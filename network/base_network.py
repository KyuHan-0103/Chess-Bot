"""
base_network.py -- Checkpoint 4: the 768 -> 1 PSQT tuner.

No hidden layer and no activation. The output IS the evaluation, in logit
units: BCEWithLogitsLoss applies the sigmoid internally, so a perfectly
fitted model emits cp/K and `output * K` is centipawns.

769 parameters, so this converges in a few thousand steps. It exists to prove
the pipeline -- parquet, SAN replay, encoding, target sign -- not to play
chess. If the printed tables look like piece-square tables, everything
upstream works.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

EPOCHS = 2
K = 174                      # sigmoid scale: 400/ln(10), the base-10 Elo curve
BATCH = 7_500
VAL_BATCH = 40_000
LR = 0.01
TARGET_ROWS = 8_000_000      # train rows to keep; 769 params need nothing like 159M
PATIENCE = 2                 # epochs without val improvement before stopping
TRAIN_DIR = "2016_fen_eval/prepped_train"
VAL_DIR = "2016_fen_eval/prepped_val"
CHECKPOINT = "best.pt"

BUCKETS = ["opp K", "opp Q", "opp R", "opp B", "opp N", "opp P",
           "my P",  "my N",  "my B",  "my R",  "my Q",  "my K"]


def show_tables(weight, K, bias=None):
    """weight: the (769, 1) accumulator weight as a numpy array."""
    table = weight[:768].reshape(12, 8, 8) * K        # logits -> centipawns
    for b, name in enumerate(BUCKETS):
        print(f"\n{name}   (row 0 = far rank from the mover, row 7 = own back rank)")
        for r in range(8):
            print("   " + " ".join(f"{v:7.0f}" for v in table[b, r]))
    if bias is not None:
        print(f"\nbias (tempo): {np.ravel(bias)[0] * K:+.0f} cp")
    print("\nmean by bucket:")
    for b, name in enumerate(BUCKETS):
        print(f"  {name:7} {table[b].mean():+9.0f} cp"
              f"   (min {table[b].min():+.0f}, max {table[b].max():+.0f})")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.accumulator = nn.EmbeddingBag(769, 1, mode='sum', padding_idx=768)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        return self.accumulator(indices) + self.bias


def load_split(folder : str, target_rows : int | None = None):
    """Read features/targets into RAM, optionally keeping every Nth row.

    mmap is the wrong tool once the loop shuffles: 7,500 random rows land on
    7,500 separate pages, so every batch becomes 7,500 disk reads and a file
    larger than RAM is re-read every epoch. Subsampling to something that fits
    turns each batch into a pure memory gather.

    Strided rather than a contiguous slice, because rows are still in game
    order -- a contiguous chunk would be one slice of the year, not a sample
    of it. Taking every Nth row also lands 3-4 positions per game instead of
    70 consecutive ones, which cuts the within-game correlation that makes the
    effective sample size so much smaller than the row count.
    """
    features = np.load(f"{folder}/features.npy", mmap_mode='r')
    targets = np.load(f"{folder}/targets.npy", mmap_mode='r')
    if len(features) != len(targets):
        raise ValueError(f"{folder}: {len(features):,} features vs {len(targets):,} targets")

    total = len(features)
    step = max(1, total // target_rows) if target_rows else 1
    # ascontiguousarray does the one pass over the file; after this it is RAM
    features = torch.from_numpy(np.ascontiguousarray(features[::step]))
    targets = torch.from_numpy(np.ascontiguousarray(targets[::step]))
    print(f"  {folder}: {total:,} rows, keeping every {step} -> {len(features):,} "
          f"({features.numel() * 2 / 1e6:.0f} MB)", flush=True)
    return features, targets


def run_epoch(model, features, targets, optimizer, loss_fn, epoch):
    """One training pass. Returns the mean loss per batch."""
    model.train()
    n = len(features)
    perm = torch.randperm(n)
    total = 0.0
    batches = 0
    started = time.monotonic()

    for start in range(0, n, BATCH):
        # sorted within the batch for cache locality; the batch is still a
        # uniformly random subset, so shuffling quality is unchanged
        idx = perm[start:start + BATCH].sort().values
        target = torch.sigmoid(targets[idx].float() / K).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(features[idx].long())          # EmbeddingBag needs int64
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        total += loss.item()
        batches += 1
        if batches % 100 == 0:
            done = start + len(idx)
            rate = done / max(time.monotonic() - started, 1e-9)
            print(f"  epoch {epoch} {done / n:5.1%}  loss {loss.item():.5f}  "
                  f"{rate:,.0f} rows/s  eta {(n - done) / rate:4.0f}s", flush=True)

    return total / max(batches, 1)


def evaluate(model, features, targets, loss_fn):
    """Mean loss per batch over the validation set."""
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(features), VAL_BATCH):
            chunk = slice(start, start + VAL_BATCH)
            target = torch.sigmoid(targets[chunk].float() / K).unsqueeze(1)
            total += loss_fn(model(features[chunk].long()), target).item()
            batches += 1
    return total / max(batches, 1)

def main():
    print("loading", flush=True)
    train_features, train_targets = load_split(TRAIN_DIR, TARGET_ROWS)
    val_features, val_targets = load_split(VAL_DIR)      # 212 MB, keep all of it

    model = Net()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\n{len(train_features) // BATCH:,} batches/epoch, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters\n", flush=True)

    best_val_loss = float('inf')
    best_weights = None
    stale = 0

    for epoch in range(EPOCHS):
        started = time.monotonic()
        train_loss = run_epoch(model, train_features, train_targets,
                               optimizer, loss_fn, epoch)
        val_loss = evaluate(model, val_features, val_targets, loss_fn)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # written every time it improves, so a crash never costs the best model
            torch.save(best_weights, CHECKPOINT)
            stale = 0
        else:
            stale += 1

        print(f"epoch {epoch}  train {train_loss:.5f}  val {val_loss:.5f}"
              f"{'  *best' if improved else ''}  "
              f"({time.monotonic() - started:.0f}s)\n", flush=True)

        if stale >= PATIENCE:
            print(f"no improvement for {PATIENCE} epochs, stopping", flush=True)
            break

    print(f"best val loss {best_val_loss:.5f}, weights in {CHECKPOINT}")
    show_tables(best_weights["accumulator.weight"].numpy(), K,
                best_weights["bias"].numpy())


if __name__ == "__main__":
    main()