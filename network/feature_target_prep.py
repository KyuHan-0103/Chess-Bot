"""
stage2.py -- fen/eval parquet parts -> features.npy + targets.npy

Two things keep this bounded. The FEN is walked once directly instead of being
handed to from_fen, which builds a whole Chess object (hash, both piece sets,
phase, king squares) that nothing here uses. And rows are written out in shards
as they are produced, then merged into one memory-mapped array, so peak memory
is one shard rather than the whole dataset.
"""
import json
import os
import time
import numpy as np
import pyarrow.dataset as ds

MIN_PLY = 12
TRAIN_DIR = "2016_fen_eval/train"
VAL_DIR = '2016_fen_eval/val'
OUT_DIR = '2016_fen_eval/prepped'
EVAL_CLAMP = 20_000
SHARD_ROWS = 2_000_000     # rows buffered before a shard is written: ~128 MB of features
REPORT_EVERY_S = 10

#Bucket per FEN letter, with the mover's pieces always in 6-11 and the
#opponent's in 0-5. Two tables instead of a branch per piece: picking the
#table once per position is the whole side-to-move flip.
_MOVER   = {'P':6, 'N':7, 'B':8, 'R':9, 'Q':10, 'K':11,
            'p':5, 'n':4, 'b':3, 'r':2, 'q':1,  'k':0}
_OPPOSED = {'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11,
            'P':5, 'N':4, 'B':3, 'R':2, 'Q':1,  'K':0}


def fen_to_indices(fen : str):
    """(feature indices, side_to_move) straight from the FEN text.

    Walks the placement field once. No board array, no Chess object, no hash,
    no piece sets -- about 3.8 us against 40 us through from_fen.

    Row 0 of the placement is rank 8, so a black-to-move position reads
    7 - row: the vertical mirror. Files are never flipped, because White's
    a-file is Black's a-file too.
    """
    placement, side = fen.split(' ', 2)[:2]
    white_to_move = side == 'w'
    table = _MOVER if white_to_move else _OPPOSED
    out = []
    row = col = 0
    for ch in placement:
        if ch == '/':
            row += 1
            col = 0
        elif ch <= '8':           # '/' is already handled, so this is a digit
            col += ord(ch) - 48
        else:
            r = row if white_to_move else 7 - row
            out.append(table[ch] * 64 + r * 8 + col)
            col += 1
    return out, (1 if white_to_move else -1)


def position_key(fen : str):
    return " ".join(fen.split()[:4])


class ShardWriter:
    """Buffer rows, write a shard every SHARD_ROWS, then merge the shards into
    one memory-mapped array.

    The merge never holds more than one shard: open_memmap creates the final
    array on disk at full size, and each shard is copied into a slice of it.
    The .tmp names mean a crash leaves no loadable features.npy behind.
    """

    def __init__(self, folder : str):
        self.folder = folder
        self.shard_dir = os.path.join(folder, '_shards')
        os.makedirs(self.shard_dir, exist_ok=True)
        self.feature_buffer = []
        self.target_buffer = []
        self.buffered = 0
        self.shards = []
        self.rows = 0
        self.clamped = 0

    def add(self, index_lists : list, evals : list) -> None:
        if not index_lists:
            return
        feats = np.full((len(index_lists), 32), 768, dtype=np.int16)
        for i, indices in enumerate(index_lists):
            feats[i, :len(indices)] = indices
        #int32 first: mate scores reach +-100000 and would wrap in int16
        targets = np.array(evals, dtype=np.int32)
        self.clamped += int(np.sum(np.abs(targets) > EVAL_CLAMP))
        targets = np.clip(targets, -EVAL_CLAMP, EVAL_CLAMP).astype(np.int16)

        self.feature_buffer.append(feats)
        self.target_buffer.append(targets)
        self.buffered += len(index_lists)
        self.rows += len(index_lists)
        if self.buffered >= SHARD_ROWS:
            self._write_shard()

    def _write_shard(self) -> None:
        if not self.buffered:
            return
        n = len(self.shards)
        feature_path = os.path.join(self.shard_dir, f'f-{n:05d}.npy')
        target_path = os.path.join(self.shard_dir, f't-{n:05d}.npy')
        np.save(feature_path, np.concatenate(self.feature_buffer, axis=0))
        np.save(target_path, np.concatenate(self.target_buffer, axis=0))
        self.shards.append((feature_path, target_path, self.buffered))
        #Dropped here rather than after the merge, so the buffer never
        #coexists with its own concatenated copy beyond one shard
        self.feature_buffer = []
        self.target_buffer = []
        self.buffered = 0

    def merge(self) -> int:
        """Concatenate the shards into features.npy / targets.npy, then remove them."""
        self._write_shard()
        total = sum(rows for _, _, rows in self.shards)
        if total == 0:
            raise RuntimeError(f'no rows produced for {self.folder}')

        feature_tmp = os.path.join(self.folder, 'features.npy.tmp')
        target_tmp = os.path.join(self.folder, 'targets.npy.tmp')
        features = np.lib.format.open_memmap(
            feature_tmp, mode='w+', dtype=np.int16, shape=(total, 32))
        targets = np.lib.format.open_memmap(
            target_tmp, mode='w+', dtype=np.int16, shape=(total,))

        at = 0
        for feature_path, target_path, rows in self.shards:
            #mmap_mode on the source too: the copy goes page cache to page
            #cache and never materialises a shard on the heap
            features[at:at + rows] = np.load(feature_path, mmap_mode='r')
            targets[at:at + rows] = np.load(target_path, mmap_mode='r')
            at += rows
        features.flush()
        targets.flush()
        del features, targets

        os.replace(feature_tmp, os.path.join(self.folder, 'features.npy'))
        os.replace(target_tmp, os.path.join(self.folder, 'targets.npy'))

        for feature_path, target_path, _ in self.shards:
            os.remove(feature_path)
            os.remove(target_path)
        os.rmdir(self.shard_dir)
        return total


class Progress:
    """One line every REPORT_EVERY_S seconds. The total comes from the parquet
    footers and is the UNFILTERED row count, so the percentage runs ahead of
    reality once the ply/eval filter starts dropping rows."""

    def __init__(self, label : str, total : int) -> None:
        self.label = label
        self.total = max(total, 1)
        self.scanned = self.kept = self.dropped = 0
        self.last = self.start = time.monotonic()

    def update(self, scanned : int, kept : int, dropped : int) -> None:
        self.scanned += scanned
        self.kept += kept
        self.dropped += dropped
        if time.monotonic() - self.last >= REPORT_EVERY_S:
            self.report()

    def report(self) -> None:
        now = time.monotonic()
        self.last = now
        elapsed = max(now - self.start, 1e-9)
        rate = max(self.scanned / elapsed, 1e-9)
        remaining = max(self.total - self.scanned, 0) / rate
        print(f"  {self.label}: {self.scanned / self.total:5.1%} | "
              f"{self.kept:,} kept | {self.dropped:,} dropped | "
              f"{rate:,.0f} rows/s | "
              f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.0f} left",
              flush=True)


def prep_train_val_data(input_dir, non_dupe_dir, data_type : str, dedupe : bool = True):
    """dedupe=True drops positions that also appear in non_dupe_dir.

    Only train is deduped. Removing the overlap from train already guarantees
    the two sets share nothing, and leaving val untouched keeps it a faithful
    sample instead of one with its most common positions removed.
    """
    out = f'{OUT_DIR}_{data_type}'
    os.makedirs(out, exist_ok=True)
    data = ds.dataset(input_dir, format='parquet')
    batches = data.to_batches(
        columns=['fen', 'eval'],
        filter=(ds.field('ply') >= MIN_PLY) & (ds.field('eval').is_valid())
    )

    dupe_keys = set()
    if dedupe:
        other = ds.dataset(non_dupe_dir, format='parquet').to_table(columns=['fen'])
        #Hashes, not the strings: at this row count the strings alone cost a
        #few hundred MB. A 64-bit collision would drop one extra train row,
        #which costs nothing.
        dupe_keys = {hash(position_key(f)) for f in other.column('fen').to_pylist()}
        del other
        print(f"  {data_type}: deduping against {len(dupe_keys):,} positions", flush=True)

    writer = ShardWriter(out)
    progress = Progress(data_type, data.count_rows())
    dropped = 0

    for batch in batches:
        fens = batch.column('fen').to_pylist()
        evals = batch.column('eval').to_numpy(zero_copy_only=False)
        batch_indices = []
        batch_evals = []
        batch_dropped = 0
        for fen, ev in zip(fens, evals):
            if dedupe and hash(position_key(fen)) in dupe_keys:
                dropped += 1
                batch_dropped += 1
                continue
            indices, side_to_move = fen_to_indices(fen)
            batch_indices.append(indices)
            #eval is stored White-POV; side_to_move flips it to match the
            #side-to-move-relative features above
            batch_evals.append(int(ev) * side_to_move)

        writer.add(batch_indices, batch_evals)
        progress.update(len(fens), len(batch_indices), batch_dropped)

    rows = writer.merge()

    meta = {
        'source': input_dir,
        'min_ply': MIN_PLY,
        'rows': rows,
        'deduped_against': non_dupe_dir if dedupe else None,
        'dropped_val_overlap': dropped,
        'eval': 'centipawns from the side to move',
        'features': 'side-to-move relative: mover in buckets 6-11, opponent in 0-5, 768 = padding',
        'eval_clamp': EVAL_CLAMP,
        'clamped_rows': writer.clamped,
    }
    with open(os.path.join(out, 'meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=2)

    progress.report()
    print(f"  {data_type}: {rows:,} rows -> {out}/ "
          f"({dropped:,} dropped, {writer.clamped:,} clamped)\n", flush=True)


def main():
    prep_train_val_data(TRAIN_DIR, VAL_DIR, "train", dedupe=True)
    prep_train_val_data(VAL_DIR, TRAIN_DIR, 'val', dedupe=False)


if __name__ == '__main__':
    main()