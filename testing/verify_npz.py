"""
verify_npz.py -- does net.npz evaluate identically to best.pt?

    python3 -m testing.verify_npz

The one test that has to pass before the engine stops importing torch. A
transposed matrix or a misread key produces a net that runs, returns plausible
centipawn numbers, and plays badly -- the hardest failure in this project to
find from game output, and the cheapest to rule out here.

Positions come from real play rather than random index lists, so every case
has two kings, a legal piece count, and a genuine side to move. Random
playouts cover openings, middlegames and thin endgames in one sweep.
"""
import random
import sys

import numpy as np
import torch

from core.board import Chess
from core.notation import from_fen, to_fen
from core.movegen import pseudo_legal_moves
from core.rules import make_move, undo_move, game_result
from network.base_network import Net, K

CHECKPOINT = "best.pt"
NPZ = "net.npz"
PLAYOUTS = 40           # random games to walk
PLIES = 60              # maximum plies per playout
TOLERANCE = 1e-3        # centipawns; float32 reassociation is ~1e-4


# ---------------------------------------------------------------------------
# the two implementations
# ---------------------------------------------------------------------------

def piece_indexer(chess):
    """Identical to NetworkEvaluator.piece_indexer -- the shared input path."""
    board, bool_flip = chess.chess_board, chess.side_to_move == -1
    indices = []
    for squares in (chess.white_pieces, chess.black_pieces):
        for r, c in squares:
            piece = board[r][c]
            piece_val = abs(piece) + 5 if (piece > 0) != bool_flip else 6 - abs(piece)
            indices.append(piece_val * 64 + (7 - r if bool_flip else r) * 8 + c)
    return indices


class TorchEval:
    def __init__(self, path=CHECKPOINT):
        self.model = Net()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()

    def __call__(self, chess):
        features = torch.tensor([piece_indexer(chess)])
        with torch.no_grad():
            return self.model(features).item() * K


class NumpyEval:
    def __init__(self, path=NPZ):
        z = np.load(path)
        self.W1 = z["accumulator.weight"]
        self.b1 = z["acc_bias"]
        self.W2 = z["l1.weight"]
        self.b2 = z["l1.bias"]
        self.W3 = z["out.weight"].ravel()
        self.b3 = float(z["out.bias"][0])

    def __call__(self, chess):
        acc = self.W1[piece_indexer(chess)].sum(axis=0)
        h = np.clip(acc + self.b1, 0, 1)
        h = np.clip(self.W2 @ h + self.b2, 0, 1)
        return float(self.W3 @ h + self.b3) * K


# ---------------------------------------------------------------------------
# positions to test on
# ---------------------------------------------------------------------------

SUITE_FENS = (
    ("start",        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
    ("kiwipete",     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"),
    ("kiwipete, b",  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq -"),
    ("thin endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"),
    ("bare kings",   "8/8/4k3/8/8/4K3/8/8 w - -"),
    ("promotions",   "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -"),
    ("dense",        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -"),
)


def playout_positions(rng, games=PLAYOUTS, plies=PLIES):
    """Walk random legal games, yielding every position along the way.

    Covers openings through endgames, both sides to move, and -- unlike a
    fixed FEN list -- piece counts all the way down as material comes off.
    """
    for _ in range(games):
        chess = Chess()
        for _ in range(plies):
            yield chess
            if game_result(chess) is not None:
                break
            moves = pseudo_legal_moves(chess)
            rng.shuffle(moves)
            for origin, move in moves:
                if make_move(chess, origin, move):
                    break
            else:
                break


# ---------------------------------------------------------------------------

def main():
    try:
        torch_eval = TorchEval()
        numpy_eval = NumpyEval()
    except FileNotFoundError as error:
        print(f"missing weights: {error}")
        print(f"generate {NPZ} first:")
        print(f"  sd = torch.load('{CHECKPOINT}')")
        print(f"  np.savez('{NPZ}', **{{k: v.cpu().numpy() for k, v in sd.items()}})")
        return 1

    # the npz must carry every tensor the model declares, under the same key
    want = set(Net().state_dict().keys())
    have = set(np.load(NPZ).files)
    if want != have:
        print(f"FAIL  key mismatch\n  only in best.pt: {sorted(want - have)}"
              f"\n  only in net.npz: {sorted(have - want)}")
        return 1
    print(f"pass  both carry the same {len(want)} tensors: {sorted(want)}")

    # padding_idx is pinned at zero and never receives a gradient
    pad = np.load(NPZ)["accumulator.weight"][768]
    print(f"{'pass' if not pad.any() else 'FAIL'}  padding row 768 is all zero "
          f"(max |w| = {np.abs(pad).max():.2e})")

    worst = 0.0
    worst_at = None
    checked = 0

    print("\nnamed positions")
    for name, fen in SUITE_FENS:
        chess = from_fen(fen)
        t, n = torch_eval(chess), numpy_eval(chess)
        gap = abs(t - n)
        checked += 1
        if gap > worst:
            worst, worst_at = gap, to_fen(chess)
        mark = "pass" if gap <= TOLERANCE else "FAIL"
        print(f"  {mark}  {name:14} torch {t:+9.2f}  numpy {n:+9.2f}  gap {gap:.2e}")

    print(f"\n{PLAYOUTS} random playouts")
    rng = random.Random(0xC0FFEE)
    for chess in playout_positions(rng):
        t, n = torch_eval(chess), numpy_eval(chess)
        gap = abs(t - n)
        checked += 1
        if gap > worst:
            worst, worst_at = gap, to_fen(chess)

    print(f"  {checked:,} positions compared")
    print(f"  worst gap {worst:.3e} cp   (tolerance {TOLERANCE})")
    if worst_at and worst > TOLERANCE:
        print(f"  at {worst_at}")

    ok = worst <= TOLERANCE and not pad.any() and want == have
    print("\n" + "=" * 62)
    print("PASS -- net.npz matches best.pt, the engine can drop torch"
          if ok else "FAIL -- do not ship this npz")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())