"""
final_tests.py -- end-to-end regression suite.

perft.py checks move generation and the board invariants. This checks the
things above that layer: the search, the transposition table, the search
budget, endgame detection, FEN and UCI notation, the arena, and the awkward
edge cases.

    python3 -m testing.final_tests
"""
import math
import sys
import time

from arena.match import (Adjudicator, DRAW, WHITE_WIN, BLACK_WIN,
                         aggregate, elo_from_score, play_match, play_one_game)
from core.board import Chess
from core.constants import MATE, PIECE_VALUES
from core.movegen import (Move, pseudo_legal_moves, legal_move_list,
                          get_castling_rights)
from core.notation import from_fen, to_fen, move_name, parse_uci_move
from core.rules import (make_move, undo_move, game_result, has_legal_move,
                        insufficient_material)
from core.zobrist import full_hash
from engine.evaluate import self_evaluate
from engine.player import RandomPlayer, SearchPlayer
from engine.search import SearchLimits, Timeout, alpha_beta_engine
from testing.perft import Failures, perft, roundtrip_diffs

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "pass" if condition else "FAIL"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def board_intact(chess, reference):
    return (chess.chess_board == reference
            and chess.zobrist == full_hash(chess)
            and chess.phase == chess.count_phase()
            and len(chess.move_order) == 0
            and chess.white_pieces == {(r, c) for r in range(8) for c in range(8)
                                       if chess.chess_board[r][c] > 0}
            and chess.black_pieces == {(r, c) for r in range(8) for c in range(8)
                                       if chess.chess_board[r][c] < 0})


SUITE_FENS = (
    ("start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"),
    ("promo", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -"),
)


# ---------------------------------------------------------------------------
def test_engine_smoke():
    """The two line test whose absence let a broken engine ship."""
    print("\nthe engine runs at all")
    engine = alpha_beta_engine()
    move, value = engine.search(Chess(), 4, time_limit=5)
    check("search() returns a move from the start position", move is not None,
          f"{move_name(*move) if move else None} score {value}")
    check("and reports the nodes it spent", engine.node_count() > 0,
          f"{engine.node_count():,} nodes")


def test_search_leaves_board_clean():
    print("\nsearch leaves the board untouched")
    for name, fen in SUITE_FENS[:1] + SUITE_FENS[1:2] + SUITE_FENS[3:]:
        chess = from_fen(fen)
        ref = [row[:] for row in chess.chess_board]
        engine = alpha_beta_engine()
        move, value = engine.get_best_move(chess, 4)
        check(f"{name}: board/hash/phase/sets/stack all restored",
              board_intact(chess, ref))
        check(f"{name}: returned a playable move", move is not None
              and make_move(chess, *move))
        if chess.move_order:
            undo_move(chess)


def test_tt_matches_plain_search():
    print("\ntransposition table returns the same score as a plain search")
    for name, fen in SUITE_FENS:
        for depth in (3, 4):
            #Each engine owns its table, so switching the probe off is now a
            #per-instance change rather than a global monkeypatch
            off_engine = alpha_beta_engine()
            off_engine.t_table.tt_probe = lambda *a, **k: (None, None)
            _, off = off_engine.get_best_move(from_fen(fen), depth)

            on_engine = alpha_beta_engine()
            _, on = on_engine.get_best_move(from_fen(fen), depth)
            check(f"{name} depth {depth}: {off} == {on}", off == on)


def test_engines_are_independent():
    print("\ntwo engines share nothing")
    a, b = alpha_beta_engine(), alpha_beta_engine()
    check("separate transposition tables", a.t_table is not b.t_table)
    check("separate underlying arrays", a.t_table.table is not b.t_table.table)

    a.get_best_move(Chess(), 4)
    filled_a = sum(1 for e in a.t_table.table if e is not None)
    filled_b = sum(1 for e in b.t_table.table if e is not None)
    check("A searching does not populate B's table",
          filled_a > 0 and filled_b == 0, f"A {filled_a}, B {filled_b}")

    #The bug this replaces: node_limit used to be a module global, so one
    #player's budget silently became the other's
    a.search(Chess(), 4, node_limit=500)
    b.search(Chess(), 4, time_limit=2.0)
    check("A's node budget does not clamp B", b.node_count() > 500,
          f"A {a.node_count()}, B {b.node_count()}")

    #And a table survives searches but not new_game()
    a.new_game()
    check("new_game() clears the table",
          all(e is None for e in a.t_table.table))


def test_time_limit():
    print("\ntime limit is respected")
    for limit in (1.0, 3.0, 8.0):
        engine = alpha_beta_engine()
        start = time.perf_counter()
        move, _ = engine.search(Chess(), 20, time_limit=limit)
        elapsed = time.perf_counter() - start
        check(f"limit {limit}s honoured (took {elapsed:.2f}s)",
              move is not None and elapsed < limit * 1.35,
              f"{elapsed / limit:.2f}x, {engine.node_count():,} nodes")

    #A deep search under a generous limit has to actually go deep. The old
    #node cap made every search stop at 2048 nodes while still "passing" the
    #timing test above, because finishing early is not a timing failure.
    engine = alpha_beta_engine()
    engine.search(Chess(), 20, time_limit=4.0)
    check("a 4s search visits far more than the poll interval",
          engine.node_count() > 20 * SearchLimits.POLL_INTERVAL,
          f"{engine.node_count():,} nodes vs poll interval "
          f"{SearchLimits.POLL_INTERVAL}")


def test_node_limit():
    print("\nnode limit is respected")
    for limit in (1000, 20000, 100000):
        engine = alpha_beta_engine()
        move, _ = engine.search(Chess(), 20, node_limit=limit)
        check(f"limit {limit:,} nodes honoured",
              move is not None and engine.node_count() <= limit,
              f"{engine.node_count():,} used")

    #No limit means no limit, not the old 2048 default
    engine = alpha_beta_engine()
    engine.search(Chess(), 5)
    check("no node_limit means unlimited",
          engine.node_count() > SearchLimits.POLL_INTERVAL,
          f"{engine.node_count():,} nodes at depth 5")

    #Same budget, same position, same answer: this is what makes a
    #node-limited match reproducible where a timed one is not
    first = alpha_beta_engine().search(Chess(), 8, node_limit=30000)
    second = alpha_beta_engine().search(Chess(), 8, node_limit=30000)
    check("a node limited search is reproducible", first == second,
          f"{first[1]} vs {second[1]}")


def test_never_returns_none_with_legal_moves():
    print("\nsearch never returns None when a legal move exists")
    dense = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -"
    for limit in (1, 5, 40, 500):
        move, _ = alpha_beta_engine().search(from_fen(dense), 6, node_limit=limit)
        check(f"node_limit {limit} still produced a move", move is not None)
    for limit in (0.0005, 0.005, 0.05):
        move, _ = alpha_beta_engine().search(Chess(), 20, time_limit=limit)
        check(f"time_limit {limit}s still produced a move", move is not None)
    #Both budgets at once, both absurd
    move, _ = alpha_beta_engine().search(Chess(), 20, node_limit=1, time_limit=0.0001)
    check("node and time limits together still produced a move", move is not None)


def test_search_limits_unit():
    print("\nSearchLimits in isolation")
    limits = SearchLimits(node_limit=10)
    raised = False
    try:
        for _ in range(20):
            limits.check()
    except Timeout:
        raised = True
    check("raises Timeout at the node ceiling", raised and limits.nodes == 10,
          f"stopped at {limits.nodes}")

    limits = SearchLimits(node_limit=10)
    limits.armed = False
    for _ in range(100):
        limits.check()
    check("unarmed limits count but never raise", limits.nodes == 100)

    limits = SearchLimits()
    for _ in range(5000):
        limits.check()
    check("no limits means never raising", limits.nodes == 5000)

    limits = SearchLimits(time_limit=-1.0)
    check("an already expired deadline is not read before the first poll",
          limits.nodes == 0)


def test_mate_and_draws():
    print("\nendgame detection")
    chess = from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -")
    move, value = alpha_beta_engine().search(chess, 4, time_limit=5)
    check("mate in 1 found", value > MATE - 1000, f"score {value}")
    make_move(chess, *move)
    check("game_result reports Checkmate", game_result(chess) == "Checkmate")

    chess = from_fen("k7/8/1Q6/8/8/8/8/K7 b - -")
    check("stalemate: no legal moves", not has_legal_move(chess))
    check("game_result reports stalemate",
          game_result(chess) == "Draw by stalemate")
    move, _ = alpha_beta_engine().search(chess, 3, time_limit=5)
    check("search returns None when mated/stalemated", move is None)

    chess = from_fen("7k/5Q2/6K1/8/8/8/8/8 b - -")
    check("queen stalemate is a draw, not a mate",
          game_result(chess) == "Draw by stalemate")

    chess = from_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - -")
    check("back-rank mate: game_result reports Checkmate",
          game_result(chess) == "Checkmate")

    chess = from_fen("7k/5Q1K/8/8/8/8/8/8 b - -")
    check("supported queen mate: game_result reports Checkmate",
          game_result(chess) == "Checkmate")

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


def test_insufficient_material():
    print("\ninsufficient material")
    #(name, fen, drawn)
    cases = (
        ("K v K", "8/8/4k3/8/8/4K3/8/8 w - -", True),
        ("K v KN", "8/8/4k3/8/8/4K3/8/5n2 w - -", True),
        ("K v KB", "8/8/4k3/8/8/4K3/8/5b2 w - -", True),
        ("KN v K", "8/8/4k3/8/8/4K3/8/5N2 w - -", True),
        #f1 is light, d5 is light: same colour, so drawn
        ("KB v KB same colour", "8/8/4k3/3b4/8/4K3/8/5B2 w - -", True),
        #f1 is light, c5 is dark: opposite colours, so not drawn
        ("KB v KB opposite", "8/8/4k3/2b5/8/4K3/8/5B2 w - -", False),
        ("K v KP", "8/8/4k3/8/8/4K3/8/5p2 w - -", False),
        ("K v KR", "8/8/4k3/8/8/4K3/8/5r2 w - -", False),
        ("K v KQ", "8/8/4k3/8/8/4K3/8/5q2 w - -", False),
        ("KBN v K", "8/8/4k3/8/8/4K3/8/4BN2 w - -", False),
        ("KNN v K", "8/8/4k3/8/8/4K3/8/4NN2 w - -", False),
        ("start position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", False),
    )
    for name, fen, drawn in cases:
        chess = from_fen(fen)
        check(f"{name}: {'drawn' if drawn else 'playable'}",
              insufficient_material(chess) == drawn)

    chess = from_fen("8/8/4k3/8/8/4K3/8/8 w - -")
    check("game_result reports it",
          game_result(chess) == "Draw by insufficient material")

    #The gate has to reject a full board before doing any real work
    check("a full board is rejected by the piece count gate",
          not insufficient_material(Chess()))


def test_move_generators_agree():
    print("\nboth move generators agree")
    for fen in [f for _, f in SUITE_FENS] + [
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -"]:
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

    chess = from_fen("8/P7/8/8/8/8/8/K6k w - -")
    try:
        make_move(chess, (1, 0), Move(0, 0, 0))
        check("promotion without a piece type raises", False)
    except ValueError:
        check("promotion without a piece type raises", True)

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

    chess = from_fen("r3k2r/8/8/8/8/8/4r3/R3K2R w KQ -")
    castles = [(o, m) for o, m in pseudo_legal_moves(chess)
               if abs(chess.chess_board[o[0]][o[1]]) == 6
               and abs(m.col - o[1]) == 2]
    check("cannot castle out of check", len(castles) == 0,
          f"{len(castles)} offered")


def test_uci_roundtrip():
    """move_name and parse_uci_move are inverses, so test them as a pair."""
    print("\nUCI notation round trip")
    for name, fen in SUITE_FENS:
        chess = from_fen(fen)
        bad = []
        for origin, move in pseudo_legal_moves(chess):
            text = move_name(origin, move)
            try:
                if parse_uci_move(text) != (origin, move):
                    bad.append(text)
            except Exception as error:
                bad.append(f"{text} ({type(error).__name__})")
        check(f"{name}: every generated move survives encode then parse",
              not bad, "; ".join(bad[:4]))

    #Written out by hand, because a round trip alone cannot catch both halves
    #being wrong in the same way. This is the exact bug that shipped: the file
    #letter went into the row and the rank digit was used unconverted.
    literal = (
        ("e2e4", (6, 4), Move(4, 4)),
        ("e7e5", (1, 4), Move(3, 4)),
        ("a1a8", (7, 0), Move(0, 0)),
        ("h8h1", (0, 7), Move(7, 7)),
        ("e1g1", (7, 4), Move(7, 6)),          # kingside castle
        ("e1c1", (7, 4), Move(7, 2)),          # queenside castle
        ("e7e8q", (1, 4), Move(0, 4, 5)),
        ("a7a8n", (1, 0), Move(0, 0, 2)),
        ("b2b1r", (6, 1), Move(7, 1, 4)),
        ("g7g8b", (1, 6), Move(0, 6, 3)),
    )
    for text, origin, move in literal:
        got = parse_uci_move(text)
        check(f"{text} parses to the right squares", got == (origin, move),
              "" if got == (origin, move) else str(got))
        check(f"{text} encodes back", move_name(origin, move) == text,
              move_name(origin, move))


def test_fen_roundtrip():
    print("\nFEN round trip")
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

    chess = Chess()
    for origin, target in (((6, 4), Move(4, 4)), ((1, 4), Move(3, 4)),
                           ((7, 6), Move(5, 5)), ((0, 1), Move(2, 2)),
                           ((7, 5), Move(4, 2)), ((1, 3), Move(3, 3)),
                           ((4, 4), Move(3, 3))):
        make_move(chess, origin, target)
    diffs = roundtrip_diffs(chess)
    check("a played position survives to_fen -> from_fen", not diffs,
          "; ".join(diffs))

    for name, fen, depth in (
            ("kiwipete", SUITE_FENS[1][1], 2),
            ("en passant / promotion", SUITE_FENS[2][1], 3),
            ("promotions and pins", SUITE_FENS[3][1], 2)):
        failures = Failures()
        perft(from_fen(fen), depth, check_fen=True, failures=failures)
        check(f"{name}: round trip holds at every node to depth {depth}",
              failures.fen_count == 0,
              failures.samples[0] if failures.samples else "")

    unusable = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    once = to_fen(from_fen(unusable))
    check("an unusable en passant square is dropped",
          once.split()[3] == "-", once)
    check("and dropping it is stable under further round trips",
          to_fen(from_fen(once)) == once, to_fen(from_fen(once)))


def test_players():
    print("\nplayers share one contract")
    for player in (RandomPlayer(seed=1, name="r"),
                   SearchPlayer(3, nodes=2000, name="s")):
        chess = Chess()
        reference = [row[:] for row in chess.chess_board]
        move = player.select_move(chess)
        check(f"{type(player).__name__}: select_move(chess) returns a move",
              move is not None)
        check(f"{type(player).__name__}: leaves the board untouched",
              board_intact(chess, reference))
        check(f"{type(player).__name__}: the move is legal",
              move is not None and make_move(chess, *move))
        if chess.move_order:
            undo_move(chess)

    #None means "no legal move", never "I gave up"
    for name, fen in (("checkmate", "R5k1/5ppp/8/8/8/8/8/6K1 b - -"),
                      ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - -")):
        for player in (RandomPlayer(seed=1), SearchPlayer(3, nodes=500)):
            check(f"{type(player).__name__} returns None at {name}",
                  player.select_move(from_fen(fen)) is None)

    #Uniform over the legal moves, and it terminates: the old version looped
    #forever at checkmate and called make_move with the wrong arity
    counts = {}
    player = RandomPlayer(seed=4)
    chess = Chess()
    for _ in range(4000):
        move = player.select_move(chess)
        counts[move] = counts.get(move, 0) + 1
    check("RandomPlayer covers every legal move", len(counts) == 20,
          f"{len(counts)} distinct")
    check("RandomPlayer is roughly uniform",
          min(counts.values()) > 4000 / 20 * 0.6, f"min {min(counts.values())}")

    #Two SearchPlayers must not share a table by accident
    a, b = SearchPlayer(4, nodes=3000), SearchPlayer(4, nodes=3000)
    check("two SearchPlayers own separate engines", a.engine is not b.engine)
    a.select_move(Chess())
    check("A's search does not fill B's table",
          all(e is None for e in b.engine.t_table.table))
    check("SearchPlayer reports a score", a.last_score() is not None)
    check("SearchPlayer reports nodes", a.last_nodes() > 0, f"{a.last_nodes():,}")


def test_adjudicator():
    print("\nadjudicator")
    adj = Adjudicator(draw_margin=15, draw_moves=3, draw_after_move=2,
                      resign_margin=900, resign_moves=3, ply_limit=20)

    #Level scores from both sides for long enough is a draw
    adj.reset()
    verdict = None
    for ply in range(20):
        adj.record(1 if ply % 2 == 0 else -1, 4)
        verdict = adj.verdict(ply + 1)
        if verdict:
            break
    check("agreed level scores adjudicate a draw",
          verdict is not None and verdict[0] == DRAW, str(verdict))

    #One side reporting a hopeless score loses
    adj.reset()
    verdict = None
    for ply in range(20):
        adj.record(1, -2000)
        verdict = adj.verdict(ply + 1)
        if verdict:
            break
    check("white reporting a lost score loses the game",
          verdict is not None and verdict[0] == BLACK_WIN, str(verdict))

    adj.reset()
    verdict = None
    for ply in range(20):
        adj.record(-1, -2000)
        verdict = adj.verdict(ply + 1)
        if verdict:
            break
    check("black reporting a lost score loses the game",
          verdict is not None and verdict[0] == WHITE_WIN, str(verdict))

    #A swing in the scores has to reset the run, or a single quiet moment
    #mid-game would end it
    adj.reset()
    for ply in range(20):
        adj.record(1 if ply % 2 == 0 else -1, 4 if ply % 4 else 500)
        if adj.verdict(ply + 1) and adj.verdict(ply + 1)[1] != "Ply limit":
            break
    check("a score swing resets the level run",
          adj.verdict(19) is None or adj.verdict(19)[1] == "Ply limit")

    #The safety net
    adj.reset()
    check("ply limit is the last resort",
          adj.verdict(20) == (DRAW, "Ply limit"), str(adj.verdict(20)))

    #A player with no opinion must not be counted as agreeing
    adj.reset()
    for ply in range(20):
        adj.record(1, None)
    check("None scores never adjudicate",
          adj.verdict(19) is None or adj.verdict(19)[1] == "Ply limit")


def test_arena():
    print("\narena")
    strong = SearchPlayer(4, nodes=4000, name="strong")
    weak = RandomPlayer(seed=11, name="weak")
    adj = Adjudicator(ply_limit=120, resign_moves=4)

    result = play_one_game(strong, weak, adjudicator=adj)
    check("a game finishes with an outcome", result.outcome is not None,
          f"{result.outcome} by {result.reason}")
    check("no errors in a clean game", result.error is None, str(result.error))
    check("the search player beat the random player", result.outcome == WHITE_WIN,
          f"outcome {result.outcome}")
    check("moves were recorded", len(result.moves) == result.plies,
          f"{len(result.moves)} moves, {result.plies} plies")
    check("the record replays", _replays(result), result.record()[:60])
    check("score_for flips with colour",
          result.score_for("strong") == 1.0 and result.score_for("weak") == 0.0)

    #Colours actually swap, and a paired match is symmetric for identical
    #players, which is the property that makes the swap worth doing
    results = list(play_match(SearchPlayer(3, nodes=1500, name="A"),
                              SearchPlayer(3, nodes=1500, name="B"),
                              games=4, adjudicator=Adjudicator(ply_limit=80)))
    whites = [r.white_name for r in results]
    check("colours alternate every game", whites == ["A", "B", "A", "B"],
          str(whites))
    summary = aggregate(results, "A", "B")
    check("identical players score 0.500", summary["score"] == 0.5,
          str(summary["score"]))
    check("aggregate counts every game",
          summary["games"] == 4
          and summary["wins"] + summary["draws"] + summary["losses"] == 4)
    check("aggregate splits by colour",
          len(summary["as_white"]) == 2 and len(summary["as_black"]) == 2)

    #An illegal move is a bug report, not a defeat
    class Cheater(RandomPlayer):
        def select_move(self, chess):
            return ((0, 0), Move(4, 4))

    bad = play_one_game(Cheater(name="cheat"), weak,
                        adjudicator=Adjudicator(ply_limit=10))
    check("an illegal move is recorded as an error", bad.error is not None,
          str(bad.error))
    check("and is not scored as a loss", bad.outcome is None)
    summary = aggregate([bad], "cheat", "weak")
    check("aggregate excludes errored games from the score",
          summary["games"] == 0 and len(summary["errors"]) == 1)

    #Elo maths
    check("0.5 is 0 elo", abs(elo_from_score(0.5)) < 1e-9)
    check("0.75 is about +191 elo", abs(elo_from_score(0.75) - 190.8) < 1.0,
          f"{elo_from_score(0.75):.1f}")
    check("a clean sweep is +inf, not a crash",
          elo_from_score(1.0) == math.inf)
    check("elo is antisymmetric",
          abs(elo_from_score(0.6) + elo_from_score(0.4)) < 1e-9)


def _replays(result):
    """A GameResult's record has to reproduce the game it describes."""
    chess = from_fen(result.start_fen)
    for text in result.moves:
        origin, move = parse_uci_move(text)
        if not make_move(chess, origin, move):
            return False
    return True


def test_selfplay():
    print("\nself-play game (real gameplay path)")
    chess = Chess()
    engine = alpha_beta_engine()
    problems = 0
    moves = 0
    for _ in range(40):
        stack_before = len(chess.move_order)
        move, value = engine.search(chess, 5, time_limit=0.4)
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
    check("phase still correct after the game",
          chess.phase == chess.count_phase())


if __name__ == "__main__":
    start = time.time()
    test_engine_smoke()
    test_search_leaves_board_clean()
    test_tt_matches_plain_search()
    test_engines_are_independent()
    test_search_limits_unit()
    test_time_limit()
    test_node_limit()
    test_never_returns_none_with_legal_moves()
    test_mate_and_draws()
    test_insufficient_material()
    test_move_generators_agree()
    test_edge_cases()
    test_uci_roundtrip()
    test_fen_roundtrip()
    test_players()
    test_adjudicator()
    test_arena()
    test_selfplay()

    assert self_evaluate(Chess()) == 0
    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed   ({time.time() - start:.0f}s)")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)