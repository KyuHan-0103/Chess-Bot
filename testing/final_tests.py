"""
final_tests.py -- end-to-end regression suite.

perft.py checks move generation and the board invariants. This checks the
things above that layer: the search, the transposition table, the clock,
endgame detection, FEN notation, and the awkward edge cases.

    python3 final_tests.py
"""
import math
import sys
import time

from engine import search
from core.board import Chess
from core.constants import MATE, PIECE_VALUES
from core.movegen import (Move, pseudo_legal_moves, legal_move_list,
                     get_castling_rights)
from core.notation import from_fen, to_fen
from testing.perft import Failures, perft, roundtrip_diffs
from core.rules import make_move, undo_move, game_result, has_legal_move
from core.zobrist import full_hash
from engine.evaluate import self_evaluate

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "pass" if condition else "FAIL"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def board_intact(chess, reference):
    return (chess.chess_board == reference
            and chess.zobrist == full_hash(chess)
            and len(chess.move_order) == 0
            and chess.white_pieces == {(r, c) for r in range(8) for c in range(8)
                                       if chess.chess_board[r][c] > 0}
            and chess.black_pieces == {(r, c) for r in range(8) for c in range(8)
                                       if chess.chess_board[r][c] < 0})


# ---------------------------------------------------------------------------
def test_search_leaves_board_clean():
    print("\nsearch leaves the board untouched")
    for name, fen, depth in (
            ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", 4),
            ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 4),
            ("promo", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 4)):
        chess = from_fen(fen)
        ref = [row[:] for row in chess.chess_board]
        search.table = [None] * search.TT_SIZE
        move, value = search.get_best_move(chess, depth)
        check(f"{name}: board/hash/sets/stack all restored",
              board_intact(chess, ref))
        check(f"{name}: returned a playable move", move is not None
              and make_move(chess, *move))
        if chess.move_order:
            undo_move(chess)


def test_tt_matches_plain_search():
    print("\ntransposition table returns the same score as a plain search")
    real_probe = search.tt_probe
    for name, fen in (
            ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
            ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"),
            ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"),
            ("promo", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -")):
        for depth in (3, 4):
            search.table = [None] * search.TT_SIZE
            search.tt_probe = lambda *a, **k: (None, None)
            _, off = search.get_best_move(from_fen(fen), depth)
            search.table = [None] * search.TT_SIZE
            search.tt_probe = real_probe
            _, on = search.get_best_move(from_fen(fen), depth)
            check(f"{name} depth {depth}: {off} == {on}", off == on)
    search.tt_probe = real_probe


def test_time_limit():
    print("\ntime limit is respected")
    for limit in (1.0, 3.0, 8.0):
        chess = Chess()
        start = time.perf_counter()
        move, _ = search.search(chess, 20, time_limit=limit)
        elapsed = time.perf_counter() - start
        check(f"limit {limit}s honoured (took {elapsed:.2f}s)",
              move is not None and elapsed < limit * 1.35,
              f"{elapsed / limit:.2f}x")

    print("\ntime limit never returns None, however tight")
    for limit in (0.0005, 0.005, 0.05):
        move, _ = search.search(Chess(), 20, time_limit=limit)
        check(f"limit {limit}s still produced a move", move is not None)


def test_mate_and_draws():
    print("\nendgame detection")
    chess = from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -")
    move, value = search.search(chess, 4, time_limit=5)
    check("mate in 1 found", value > MATE - 1000, f"score {value}")
    make_move(chess, *move)
    check("game_result reports Checkmate", game_result(chess) == "Checkmate")

    chess = from_fen("k7/8/1Q6/8/8/8/8/K7 b - -")
    check("stalemate: no legal moves", not has_legal_move(chess))
    check("game_result reports stalemate",
          game_result(chess) == "Draw by stalemate")
    move, _ = search.search(chess, 3, time_limit=5)
    check("search returns None when mated/stalemated", move is None)

    # 7k/5Q2/6K1 is the classic queen STALEMATE, not a mate: Qf7 does not
    # attack h8. Use a genuine mate instead.
    chess = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - -")
    check("queen stalemate is a draw, not a mate",
          game_result(chess) == "Draw by stalemate")

    chess = from_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - -")
    check("back-rank mate: game_result reports Checkmate",
          game_result(chess) == "Checkmate")

    chess = from_fen("7k/5Q1K/8/8/8/8/8/8 b - -")
    check("supported queen mate: game_result reports Checkmate",
          game_result(chess) == "Checkmate")

    # Knight shuffle back to the start position three times
    chess = Chess()
    shuffle = [((7, 6), Move(5, 5)), ((0, 6), Move(2, 5)),
               ((5, 5), Move(7, 6)), ((2, 5), Move(0, 6))]
    stopped = None
    for _ in range(3):
        for origin, target in shuffle:
            if not make_move(chess, origin, target):
                stopped = chess.game
                break
        if stopped:
            break
    check("threefold repetition detected",
          stopped == "Draw by repetition" or chess.game == "Draw by repetition",
          str(chess.game))


def test_move_generators_agree():
    print("\nboth move generators agree")
    for fen in ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
                "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -"):
        chess = from_fen(fen)
        scan = sorted((o, tuple(m)) for o, m in legal_move_list(chess))
        sets = sorted((o, tuple(m)) for o, m in pseudo_legal_moves(chess))
        check(f"{fen[:26]}... ({len(scan)} moves)", scan == sets)


def test_edge_cases():
    print("\nedge cases")
    check("get_castling_rights(side=0) returns a list, not None",
          isinstance(get_castling_rights(Chess(), 0), list),
          repr(get_castling_rights(Chess(), 0)))

    chess = Chess()
    try:
        make_move(chess, (4, 4), Move(3, 4))
        check("moving from an empty square raises", False)
    except ValueError:
        check("moving from an empty square raises", True)

    # A promotion move with no piece type must be rejected, not silently queened
    chess = from_fen("8/P7/8/8/8/8/8/K6k w - -")
    try:
        make_move(chess, (1, 0), Move(0, 0, 0))
        check("promotion without a piece type raises", False)
    except ValueError:
        check("promotion without a piece type raises", True)

    # All four promotion pieces reachable, and each undoes cleanly
    chess = from_fen("8/P7/8/8/8/8/8/K6k w - -")
    ref = [row[:] for row in chess.chess_board]
    ok = True
    for piece in (2, 3, 4, 5):
        if not make_move(chess, (1, 0), Move(0, 0, piece)):
            ok = False
            break
        if chess.chess_board[0][0] != piece:
            ok = False
        undo_move(chess)
        if not board_intact(chess, ref):
            ok = False
    check("all four promotions apply and undo cleanly", ok)

    # En passant, made and unmade
    chess = from_fen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6")
    ref = [row[:] for row in chess.chess_board]
    eps = [(o, m) for o, m in pseudo_legal_moves(chess)
           if abs(chess.chess_board[o[0]][o[1]]) == 1
           and m.col != o[1] and chess.chess_board[m.row][m.col] == 0]
    check("en passant capture is generated", len(eps) > 0, f"{len(eps)} found")
    if eps:
        make_move(chess, *eps[0])
        captured_gone = chess.chess_board[3][5] == 0
        undo_move(chess)
        check("en passant removes the right pawn", captured_gone)
        check("en passant undoes cleanly", board_intact(chess, ref))

    # Castling both ways, made and unmade
    chess = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -")
    ref = [row[:] for row in chess.chess_board]
    castles = [(o, m) for o, m in pseudo_legal_moves(chess)
               if abs(chess.chess_board[o[0]][o[1]]) == 6
               and abs(m.col - o[1]) == 2]
    check("both castles generated", len(castles) == 2, f"{len(castles)} found")
    ok = True
    for origin, target in castles:
        if not make_move(chess, origin, target):
            ok = False
            continue
        undo_move(chess)
        if not board_intact(chess, ref):
            ok = False
    check("castling undoes cleanly (board, hash, sets)", ok)

    # Cannot castle out of, through, or into check
    chess = from_fen("r3k2r/8/8/8/8/8/4r3/R3K2R w KQ -")
    castles = [(o, m) for o, m in pseudo_legal_moves(chess)
               if abs(chess.chess_board[o[0]][o[1]]) == 6
               and abs(m.col - o[1]) == 2]
    check("cannot castle out of check", len(castles) == 0,
          f"{len(castles)} offered")


def test_fen_roundtrip():
    print("\nFEN round trip")

    #Canonical FENs: all six fields, and any en passant square is one a pawn
    #can actually use, so to_fen has to give the string back unchanged
    canonical = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 3 9",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 4",
        "8/P7/8/8/8/8/8/K6k b - - 12 37",
        "4k3/8/8/8/8/8/8/4K3 w - - 99 150",
    )
    for fen in canonical:
        got = to_fen(from_fen(fen))
        check(f"unchanged: {fen[:30]}...", got == fen, "" if got == fen else got)

    check("Chess() writes the start position", to_fen(Chess()) == canonical[0],
          to_fen(Chess()))

    #The other direction, on positions that were played rather than parsed:
    #from_fen(to_fen(pos)) has to be the same position, field for field
    chess = Chess()
    for origin, target in (((6, 4), Move(4, 4)), ((1, 4), Move(3, 4)),
                           ((7, 6), Move(5, 5)), ((0, 1), Move(2, 2)),
                           ((7, 5), Move(4, 2)), ((1, 3), Move(3, 3)),
                           ((4, 4), Move(3, 3))):
        make_move(chess, origin, target)
    diffs = roundtrip_diffs(chess)
    check("a played position survives to_fen -> from_fen", not diffs,
          "; ".join(diffs))

    #Every node of a shallow tree, which is where the fiddly cases live:
    #en passant appearing and expiring, castling rights being lost,
    #promotions, and the half move clock resetting on captures
    for name, fen, depth in (
            ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 2),
            ("en passant / promotion", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 3),
            ("promotions and pins", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 2)):
        failures = Failures()
        perft(from_fen(fen), depth, check_fen=True, failures=failures)
        check(f"{name}: round trip holds at every node to depth {depth}",
              failures.fen_count == 0,
              failures.samples[0] if failures.samples else "")

    #Documented asymmetry: an en passant square is only stored when a capture
    #is available, so a FEN offering one no pawn can use is rewritten as "-".
    #Rewritten once, then stable -- to_fen output is always a fixed point
    unusable = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    once = to_fen(from_fen(unusable))
    check("an unusable en passant square is dropped",
          once.split()[3] == "-", once)
    check("and dropping it is stable under further round trips",
          to_fen(from_fen(once)) == once, to_fen(from_fen(once)))


def test_selfplay():
    print("\nself-play game (real gameplay path)")
    chess = Chess()
    problems = 0
    moves = 0
    for _ in range(40):
        stack_before = len(chess.move_order)
        move, value = search.search(chess, 5, time_limit=0.4)
        if chess.zobrist != full_hash(chess):
            problems += 1
        if len(chess.move_order) != stack_before:
            problems += 1
        if move is None:
            break
        if not make_move(chess, *move):
            problems += 1
            break
        moves += 1
        if game_result(chess) is not None:
            break
    truth_w = {(r, c) for r in range(8) for c in range(8)
               if chess.chess_board[r][c] > 0}
    check(f"{moves} moves played with no problems", problems == 0)
    check("piece sets still correct after the game",
          chess.white_pieces == truth_w)
    check("hash still correct after the game",
          chess.zobrist == full_hash(chess))


if __name__ == "__main__":
    start = time.time()
    test_search_leaves_board_clean()
    test_tt_matches_plain_search()
    test_time_limit()
    test_mate_and_draws()
    test_move_generators_agree()
    test_edge_cases()
    test_fen_roundtrip()
    test_selfplay()

    assert self_evaluate(Chess()) == 0
    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.time() - start:.0f}s)")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)