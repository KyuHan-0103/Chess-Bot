"""
perft.py -- correctness harness for the engine.

Perft ("performance test") counts the leaf nodes in the legal move tree to a
given depth. The counts below are known-correct published values, so any
mismatch means a move generation bug -- and the divided mode tells you exactly
which move's subtree is wrong.

Usage:
    python3 perft.py                    # run the standard suite
    python3 perft.py --hash             # also verify zobrist consistency
    python3 perft.py --state            # also verify make/undo restores state
    python3 perft.py --all              # every check
    python3 perft.py --divide 3         # per-root-move counts, start position
    python3 perft.py --divide 3 --fen "<FEN>"
    python3 perft.py --depth 5          # start position only, to depth 5

Debugging a mismatch:
    1. Run --divide at the shallowest failing depth.
    2. Compare against a reference (any known-good engine, or an online perft
       calculator) to find which root move has the wrong count.
    3. Make that move, then --divide the resulting position one depth shallower.
    4. Repeat until you reach depth 1, where the wrong move list is visible.
"""

import argparse
import sys
import time

from core.board import Chess
from core.constants import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
from core.movegen import Move, legal_move_list, pseudo_legal_moves
from core.rules import make_move, undo_move
from core.zobrist import full_hash

# ---------------------------------------------------------------------------
# FEN parsing
# ---------------------------------------------------------------------------

FEN_PIECES = {"p": PAWN, "n": KNIGHT, "b": BISHOP,
              "r": ROOK, "q": QUEEN, "k": KING}


def from_fen(fen):
    """Build a Chess object from a FEN string."""
    parts = fen.split()
    if len(parts) < 3:
        raise ValueError(f"FEN needs at least 3 fields, got {len(parts)}: {fen!r}")

    placement, side_field, castle_field = parts[0], parts[1], parts[2]
    ep_field = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 else 0

    board = [[0] * 8 for _ in range(8)]
    row = 0
    col = 0
    for ch in placement:
        if ch == "/":
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            piece = FEN_PIECES[ch.lower()]
            board[row][col] = piece if ch.isupper() else -piece
            col += 1

    chess = Chess()
    chess.chess_board = board
    chess.side_to_move = 1 if side_field == "w" else -1

    chess.white_king_castle = "K" in castle_field
    chess.white_queen_castle = "Q" in castle_field
    chess.black_king_castle = "k" in castle_field
    chess.black_queen_castle = "q" in castle_field

    # Locate the kings rather than trusting a hardcoded square
    chess.white_king_pos = None
    chess.black_king_pos = None
    for r in range(8):
        for c in range(8):
            if board[r][c] == KING:
                chess.white_king_pos = (r, c)
            elif board[r][c] == -KING:
                chess.black_king_pos = (r, c)
    if chess.white_king_pos is None or chess.black_king_pos is None:
        raise ValueError("FEN is missing a king")

    # This engine stores en passant as {capturing_pawn_square: destination Move}
    chess.en_passant = {}
    if ep_field != "-":
        ep_col = ord(ep_field[0]) - ord("a")
        ep_row = 8 - int(ep_field[1])
        side = chess.side_to_move
        pawn_row = ep_row + side          # square the capturing pawn sits on
        if 0 <= pawn_row < 8:
            for delta in (-1, 1):
                c = ep_col + delta
                if 0 <= c < 8 and board[pawn_row][c] == PAWN * side:
                    chess.en_passant[(pawn_row, c)] = Move(ep_row, ep_col)

    chess.half_move_clock = halfmove
    chess.position_counts = {}
    chess.game = None
    chess.play = 0
    chess.full_move_count = 1
    chess.move_order.clear()
    chess.zobrist = full_hash(chess)
    #The piece sets were built from the start position in __init__
    chess.white_pieces = {(r, c) for r in range(8) for c in range(8)
                          if board[r][c] > 0}
    chess.black_pieces = {(r, c) for r in range(8) for c in range(8)
                          if board[r][c] < 0}
    #After the piece sets, because it counts over them
    chess.phase = chess.count_phase()
    return chess


def square_name(row, col):
    return "abcdefgh"[col] + str(8 - row)


def move_name(origin, move):
    name = square_name(*origin) + square_name(move.row, move.col)
    if move.promotion:
        name += "nbrq"[[KNIGHT, BISHOP, ROOK, QUEEN].index(move.promotion)]
    return name


# ---------------------------------------------------------------------------
# State snapshot, for verifying that undo_move restores everything
# ---------------------------------------------------------------------------

def snapshot(chess):
    return (
        tuple(tuple(row) for row in chess.chess_board),
        chess.side_to_move,
        chess.white_king_pos, chess.black_king_pos,
        chess.white_king_castle, chess.white_queen_castle,
        chess.black_king_castle, chess.black_queen_castle,
        tuple(sorted(chess.en_passant.items())),
        chess.half_move_clock, chess.play, chess.full_move_count,
        chess.game, chess.zobrist,
        len(chess.move_order),
        tuple(sorted(chess.position_counts.items())),
        tuple(sorted(chess.white_pieces)),
        tuple(sorted(chess.black_pieces)),
    )


SNAPSHOT_FIELDS = (
    "chess_board", "side_to_move", "white_king_pos", "black_king_pos",
    "white_king_castle", "white_queen_castle", "black_king_castle",
    "black_queen_castle", "en_passant", "half_move_clock", "play",
    "full_move_count", "game", "zobrist", "len(move_order)", "position_counts",
    "white_pieces", "black_pieces",
)


def describe_diff(before, after):
    out = []
    for name, a, b in zip(SNAPSHOT_FIELDS, before, after):
        if a != b:
            out.append(f"{name}: {a!r} -> {b!r}")
    return out


# ---------------------------------------------------------------------------
# Perft
# ---------------------------------------------------------------------------

class Failures:
    """Collects the first few problems so output stays readable."""

    def __init__(self, limit=5):
        self.limit = limit
        self.hash_count = 0
        self.state_count = 0
        self.sets_count = 0
        self.samples = []

    def hash_bad(self, chess, path):
        self.hash_count += 1
        if len(self.samples) < self.limit:
            self.samples.append(
                f"  HASH after {' '.join(path)}\n"
                f"    incremental = {chess.zobrist}\n"
                f"    full_hash   = {full_hash(chess)}"
            )

    def sets_bad(self, chess, path, detail):
        self.sets_count += 1
        if len(self.samples) < self.limit:
            self.samples.append(
                f"  PIECE SETS after {' '.join(path)}\n      {detail}")

    def state_bad(self, path, diffs):
        self.state_count += 1
        if len(self.samples) < self.limit:
            joined = "\n      ".join(diffs)
            self.samples.append(
                f"  UNDO did not restore state after {' '.join(path)}\n"
                f"      {joined}"
            )


def perft(chess, depth, check_hash=False, check_state=False,
          check_sets=False, failures=None, path=None):
    if depth == 0:
        return 1
    if path is None:
        path = []

    total = 0
    for origin, move in legal_move_list(chess):
        before = snapshot(chess) if check_state else None

        if not make_move(chess, origin, move):
            continue

        path.append(move_name(origin, move))

        # Perft counts legal moves, not draw claims. Clearing this keeps a
        # threefold repetition deep in the tree from blocking further moves.
        # undo_move restores the saved value, so this is safe.
        chess.game = None

        if check_hash and chess.zobrist != full_hash(chess):
            failures.hash_bad(chess, path)

        if check_sets:
            board = chess.chess_board
            true_w = {(r, c) for r in range(8) for c in range(8)
                      if board[r][c] > 0}
            true_b = {(r, c) for r in range(8) for c in range(8)
                      if board[r][c] < 0}
            if chess.white_pieces != true_w or chess.black_pieces != true_b:
                failures.sets_bad(chess, path,
                    f"white missing {sorted(true_w - chess.white_pieces)} "
                    f"stale {sorted(chess.white_pieces - true_w)}; "
                    f"black missing {sorted(true_b - chess.black_pieces)} "
                    f"stale {sorted(chess.black_pieces - true_b)}")
            scan = sorted((o, tuple(m)) for o, m in legal_move_list(chess))
            sets = sorted((o, tuple(m)) for o, m in pseudo_legal_moves(chess))
            if scan != sets:
                failures.sets_bad(chess, path,
                    f"legal_move_list has {len(scan)} moves but "
                    f"pseudo_legal_moves has {len(sets)}")

        total += perft(chess, depth - 1, check_hash, check_state,
                       check_sets, failures, path)
        undo_move(chess)
        path.pop()

        if check_state:
            diffs = describe_diff(before, snapshot(chess))
            if diffs:
                failures.state_bad(path + [move_name(origin, move)], diffs)

    return total


def divide(chess, depth):
    """Node count per root move. The tool for locating a mismatch."""
    results = []
    for origin, move in legal_move_list(chess):
        if not make_move(chess, origin, move):
            continue
        chess.game = None
        count = perft(chess, depth - 1)
        undo_move(chess)
        results.append((move_name(origin, move), count))
    results.sort()
    return results


# ---------------------------------------------------------------------------
# Test suite: the standard published positions
# ---------------------------------------------------------------------------

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

SUITE = [
    ("Start position", START,
     {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609}),

    ("Kiwipete (castling, pins)",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
     {1: 48, 2: 2039, 3: 97862, 4: 4085603}),

    ("Endgame (en passant, promotion)",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
     {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624}),

    ("Promotions and pins",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
     {1: 6, 2: 264, 3: 9467, 4: 422333}),

    ("Castling edge cases",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
     {1: 44, 2: 1486, 3: 62379, 4: 2103487}),

    ("Dense middlegame",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -",
     {1: 46, 2: 2079, 3: 89890, 4: 3894594}),
]


def run_suite(max_depth, check_hash, check_state, check_sets, time_budget):
    passed = failed = skipped = 0
    all_failures = Failures()

    for name, fen, expected in SUITE:
        print(f"\n{name}")
        print(f"  {fen}")
        for depth in sorted(expected):
            if depth > max_depth:
                skipped += 1
                continue
            chess = from_fen(fen)
            start = time.time()
            got = perft(chess, depth, check_hash, check_state, check_sets,
                        all_failures)
            elapsed = time.time() - start
            want = expected[depth]

            if got == want:
                passed += 1
                mark = "pass"
                note = ""
            else:
                failed += 1
                mark = "FAIL"
                note = f"   off by {got - want:+d}"

            nps = got / elapsed if elapsed > 0 else 0
            print(f"    {mark}  depth {depth}: {got:>10,} "
                  f"(want {want:>10,}) [{elapsed:6.2f}s, {nps:>9,.0f} nps]{note}")

            if elapsed > time_budget:
                print(f"    ...stopping this position, "
                      f"depth {depth} took over {time_budget}s")
                break

    print("\n" + "=" * 66)
    print(f"move generation: {passed} passed, {failed} failed, {skipped} skipped")

    if check_hash:
        if all_failures.hash_count:
            print(f"zobrist:         {all_failures.hash_count:,} nodes where the "
                  f"incremental hash != full_hash")
        else:
            print("zobrist:         consistent at every node")

    if check_state:
        if all_failures.state_count:
            print(f"make/undo:       {all_failures.state_count:,} nodes where undo "
                  f"did not restore state")
        else:
            print("make/undo:       state fully restored at every node")

    if check_sets:
        if all_failures.sets_count:
            print(f"piece sets:      {all_failures.sets_count:,} nodes where the "
                  f"sets disagreed with the board")
        else:
            print("piece sets:      match the board at every node")

    if all_failures.samples:
        print("\nfirst few problems:")
        for sample in all_failures.samples:
            print(sample)

    print("=" * 66)
    return (failed == 0 and not all_failures.hash_count
            and not all_failures.state_count and not all_failures.sets_count)


def main():
    parser = argparse.ArgumentParser(description="Perft harness")
    parser.add_argument("--depth", type=int, default=4,
                        help="max depth to attempt per position (default 4)")
    parser.add_argument("--fen", default=START, help="position for --divide")
    parser.add_argument("--divide", type=int, metavar="D",
                        help="print per-root-move counts at depth D")
    parser.add_argument("--hash", action="store_true",
                        help="verify incremental zobrist against full_hash")
    parser.add_argument("--state", action="store_true",
                        help="verify undo_move restores every field")
    parser.add_argument("--sets", action="store_true",
                        help="verify piece sets match the board")
    parser.add_argument("--all", action="store_true",
                        help="same as --hash --state --sets")
    parser.add_argument("--budget", type=float, default=30.0,
                        help="seconds before skipping deeper depths (default 30)")
    args = parser.parse_args()

    check_hash = args.hash or args.all
    check_state = args.state or args.all
    check_sets = args.sets or args.all

    if args.divide:
        chess = from_fen(args.fen)
        print(f"divide depth {args.divide}")
        print(f"  {args.fen}\n")
        results = divide(chess, args.divide)
        for name, count in results:
            print(f"  {name:6} {count:>10,}")
        print(f"\n  {len(results)} moves, {sum(c for _, c in results):,} nodes")
        return 0

    ok = run_suite(args.depth, check_hash, check_state, check_sets, args.budget)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())   