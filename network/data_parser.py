import hashlib
import numbers
import time
import traceback
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from network.san_parser import san_text_prep_with_eval, san_make_move
from core.board import Chess
from core.notation import to_fen

INPUT_PATH = '2016_chess_eval.parquet'
OUTPUT_DIR = Path('2016_fen_eval')   # gets train/ and val/ folders of part files
SCHEMA = pa.schema([
        ('fen', pa.string()),
        ('eval', pa.int32()),
        ('ply', pa.int16()),
        ('result', pa.string())
    ])
N_EXAMPLES = 10            # number of failures to print and log in full
VAL_FRACTION = 0.02        # share of *games* (not positions) that go to val
MAX_FAIL_RATE = 0.001      # above this, assume a parser bug and stop the run
CHECK_RATE_AFTER = 5_000   # games to process before judging the failure rate
REPORT_EVERY_S = 30        # seconds between progress lines


class PartWriter:
    """Buffer rows and write them to a folder of parquet part files.

    Every `flush_every` rows become one row group, and every `rows_per_part` rows
    start a new part. The part in progress is named `_part-NNNNN.parquet.tmp` and
    renamed once it's closed, so every `part-*.parquet` file is complete, and
    pd.read_parquet(folder) skips the one still being written.
    """

    def __init__(self, folder: Path, flush_every: int = 200_000, rows_per_part: int = 5_000_000):
        folder.mkdir(parents=True, exist_ok=True)
        self._folder = folder
        self._flush_every = flush_every
        self._rows_per_part = rows_per_part
        self._rows: list[tuple] = []
        self._writer = None
        self._part = 0
        self._rows_in_part = 0
        self.rows_written = 0

    def add(self, rows: list[tuple]) -> None:
        self._rows.extend(rows)
        if len(self._rows) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Write everything in self._rows to the current part as one row group"""
        if not self._rows:
            return
        rows, self._rows = self._rows, []
        columns = {name: list(col) for name, col in zip(SCHEMA.names, zip(*rows))}
        chunk = pa.table(columns, schema=SCHEMA)
        if self._writer is None:  # opened here, not in __init__, so there's never an empty part
            self._writer = pq.ParquetWriter(str(self._tmp_path()), SCHEMA)
        self._writer.write_table(chunk)
        self._rows_in_part += len(rows)
        self.rows_written += len(rows)
        if self._rows_in_part >= self._rows_per_part:
            self._finish_part()

    def _tmp_path(self) -> Path:
        return self._folder / f'_part-{self._part:05d}.parquet.tmp'

    def _finish_part(self) -> None:
        self._writer.close()  # writes the footer; only now is the file valid
        self._tmp_path().rename(self._folder / f'part-{self._part:05d}.parquet')
        self._writer = None
        self._part += 1
        self._rows_in_part = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._writer is not None:
                self._finish_part()

    def __enter__(self) -> "PartWriter":
        return self

    def __exit__(self, *exc) -> None:
        # Runs even if the loop raises (or you hit Ctrl-C), so whatever was
        # added so far still ends up in a valid file.
        self.close()


class Monitor:
    """Progress lines, failure bookkeeping, and the early stop"""

    def __init__(self, total_games: int, failure_log: Path) -> None:
        self.total_games: int = max(total_games, 1)
        self.failure_log: Path = failure_log
        self.failures: int = 0
        self.failure_types: Counter[str] = Counter()
        self.games: int = 0
        self.positions: int = 0
        self.last_report = self.start = time.monotonic()

    def ok(self, n_positions: int) -> None:
        self.games += 1
        self.positions += n_positions
        self._check()

    def failed(self, movetext: str, err: Exception) -> None:
        self.games += 1
        self.failures += 1
        self.failure_types[type(err).__name__] += 1
        if self.failures <= N_EXAMPLES:
            summary = f"{type(err).__name__}: {err}"
            print(f"FAILED  {summary}", flush=True)
            with open(self.failure_log, "a") as f:
                # format_exc() works here because we're still inside main's except block
                f.write(f"{summary}\n{traceback.format_exc()}{movetext}\n\n")
        self._check()

    def _check(self) -> None:
        if (self.failures and self.games >= CHECK_RATE_AFTER
                and self.failures > MAX_FAIL_RATE * self.games):
            raise RuntimeError(
                f"{self.failures:,} of {self.games:,} games failed "
                f"({self.failures / self.games:.2%}), which points to a parser bug "
                f"rather than bad data. By type: {dict(self.failure_types)}. "
                f"Examples: {self.failure_log}"
            )
        if time.monotonic() - self.last_report >= REPORT_EVERY_S:
            self.report()

    def report(self) -> None:
        now = time.monotonic()
        self.last_report = now
        speed = self.games / max(now - self.start, 1e-9)
        eta = timedelta(seconds=round((self.total_games - self.games) / speed)) if speed else "?"
        print(
            f"{self.games / self.total_games:6.1%} | {self.games:,} games | "
            f"{self.positions:,} positions | {self.failures:,} failed "
            f"({self.failures / max(self.games, 1):.3%}) | {speed:,.0f} games/s | ETA {eta}",
            flush=True,
        )


def split_of(movetext: str) -> str:
    """Send a whole game to train or val by hashing its movetext.

    The same game lands in the same split on every run, in any order, and all
    of its positions go with it.
    """
    digest = hashlib.blake2b(movetext.encode(), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 10_000
    return "val" if bucket < VAL_FRACTION * 10_000 else "train"
    

def expand_game(source_moves: str) -> list[tuple[str, int | None, int]]:
    """Play through one game; return (fen, eval, ply) for every position in it"""
    chess_game = Chess(False)
    fen_eval_list: list[tuple[str, int | None, int]] = []

    for move, eval_score in san_text_prep_with_eval(source_moves):
        san_make_move(chess_game, move)
        if eval_score is not None and not isinstance(eval_score, numbers.Integral):
            raise TypeError(f'eval {eval_score!r} at ply {chess_game.play} is not an int')
        fen_eval_list.append((to_fen(chess_game), eval_score, chess_game.play))
    return fen_eval_list


def main() -> None:
    source = pq.ParquetFile(INPUT_PATH)
    total_games = source.metadata.num_rows

    if OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR}/ already exists. Delete it (or change OUTPUT_DIR) "
                         "so files from an old run can't mix with this one.")
    OUTPUT_DIR.mkdir(parents=True)
    monitor = Monitor(total_games, failure_log=OUTPUT_DIR / 'failed_games.txt')
    print(f"Expanding {total_games:,} games into {OUTPUT_DIR}/", flush=True)

    with (
        PartWriter(OUTPUT_DIR / 'train') as train,
        PartWriter(OUTPUT_DIR / 'val') as val,
    ):
        writers = {'train': train, 'val': val}
        for batch in source.iter_batches(batch_size=10_000, columns=['movetext', 'Result']):
            movetexts = batch.column('movetext').to_pylist()
            results = batch.column('Result').to_pylist()
            for game, result in zip(movetexts, results):
                try:
                    split = split_of(game)  # decided before the game is expanded
                    fen_eval_list = expand_game(game)
                except Exception as err:
                    monitor.failed(game, err)
                    continue
                writers[split].add([(fen, eval_score, ply, result)
                                    for fen, eval_score, ply in fen_eval_list])
                monitor.ok(len(fen_eval_list))

    (OUTPUT_DIR / '_SUCCESS').touch()  # only reached if every game was processed
    monitor.report()
    print(f"train: {train.rows_written:,} positions | val: {val.rows_written:,} positions")
    if monitor.failures:
        print(f"Failures by type: {dict(monitor.failure_types)} "
              f"(first {N_EXAMPLES} logged in {monitor.failure_log})")


if __name__ == '__main__':
    main()