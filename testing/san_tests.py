"""
san_tests.py -- regression suite for the SAN parser.

perft.py checks move generation. final_tests.py checks the search and the
arena. This checks the layer that turns PGN movetext into moves, which is the
one piece of the training pipeline with no other safety net: a SAN bug does not
crash, it silently feeds the trainer the wrong position.

    python3 -m testing.san_tests

The design principle throughout: cross-check SAN against notation.parse_uci_move,
which perft already verifies at every node. Every game below is written twice,
once in SAN and once in UCI. Replaying both and comparing the position after
every single ply means a SAN bug cannot hide behind a correct final position.
"""
import sys
import traceback

from core.board import Chess
from core.constants import MATE
from core.movegen import Move, legal_move_list
from core.notation import from_fen, to_fen, parse_uci_move, square_name
from core.rules import make_move, undo_move
from network.san_parser import san_make_move, san_text_prep_with_eval, eval_to_number

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "pass" if condition else "FAIL"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def attempt(name, fn, *args, **kwargs):
    """Run fn, reporting an exception as a failure rather than aborting the suite."""
    try:
        return True, fn(*args, **kwargs)
    except Exception as error:
        check(name, False, f"{type(error).__name__}: {error}")
        return False, None


def sq(text):
    """'e4' -> (row, col) in this engine's indexing (row 0 is rank 8)."""
    return (8 - int(text[1]), "abcdefgh".index(text[0]))


# ---------------------------------------------------------------------------
# 1. Tokenising movetext into (move, eval) pairs
# ---------------------------------------------------------------------------

def moves_of(pairs):
    return [pair[0] for pair in pairs]


def test_text_prep():
    """
    Two contracts at once. Every move string returned must be something
    san_make_move can consume, and every eval must come back as a number on a
    single consistent scale.
    """
    print("\ntokenising movetext into (move, eval) pairs")
    #Evals are centipawns from White's side, the unit the squash in 3.5 expects
    cases = (
        ("plain pair",
         "1. e4 { [%eval 0.17] }",
         [("e4", 17)]),
        ("negative eval",
         "1. h4 { [%eval -0.42] }",
         [("h4", -42)]),
        ("dead level",
         "1. e4 { [%eval 0.0] }",
         [("e4", 0)]),
        ("clock after eval",
         "1. e4 { [%eval 0.17] [%clk 0:03:00] }",
         [("e4", 17)]),
        ("clock before eval",
         "1. e4 { [%clk 0:03:00] [%eval 0.17] }",
         [("e4", 17)]),
        ("black continuation numbering",
         "1. e4 { [%eval 0.1] } 1... e5 { [%eval 0.2] }",
         [("e4", 10), ("e5", 20)]),
        ("annotation suffixes stripped",
         "1. d4?! { [%eval -0.1] } 1... d5!? { [%eval 0.0] } 2. Nf3! { [%eval 0.3] }",
         [("d4", -10), ("d5", 0), ("Nf3", 30)]),
        ("check suffix stripped",
         "1. Qh5+ { [%eval 2.1] }",
         [("Qh5", 210)]),
        ("result token is not a move",
         "1. e4 { [%eval 0.1] } 1... e5 { [%eval 0.2] } 1-0",
         [("e4", 10), ("e5", 20)]),
        ("comment spanning a newline",
         "1. e4 { [%eval 0.17]\n[%clk 0:03:00] } 1... e5 { [%eval 0.2] }",
         [("e4", 17), ("e5", 20)]),
        ("no space after the move number",
         "1.e4 { [%eval 0.17] } 1...e5 { [%eval 0.2] }",
         [("e4", 17), ("e5", 20)]),
        ("an eval float math gets wrong: 0.57 * 100 is 56.99999999999999",
         "1. e4 { [%eval 0.57] }",
         [("e4", 57)]),
        #'#0' has no sign of its own, so these check the parser works out who
        #delivered the mate from the move count, once for each side
        ("#0 after white's move (scholar's mate) is +MATE",
         "1. e4 { [%eval 0.2] } 1... e5 { [%eval 0.2] } 2. Qh5 { [%eval 0.0] } "
         "2... Nc6 { [%eval 0.0] } 3. Bc4 { [%eval -0.1] } 3... Nf6?? { [%eval #1] } "
         "4. Qxf7# { [%eval #0] } 1-0",
         [("e4", 20), ("e5", 20), ("Qh5", 0), ("Nc6", 0), ("Bc4", -10),
          ("Nf6", MATE - 1), ("Qxf7", MATE)]),
        ("#0 after black's move (fool's mate) is -MATE",
         "1. f3 { [%eval -0.6] } 1... e5 { [%eval -0.3] } 2. g4?? { [%eval #-1] } "
         "2... Qh4# { [%eval #0] } 0-1",
         [("f3", -60), ("e5", -30), ("g4", -(MATE - 1)), ("Qh4", -MATE)]),
    )
    for name, text, want in cases:
        ok, got = attempt(f"text_prep: {name}", san_text_prep_with_eval, text)
        if ok:
            check(f"text_prep: {name}", got == want, f"got {got!r}")


def test_eval_conversion():
    """
    eval_to_number is where a sign error would be invisible. A mate score has
    to keep the side it favours, and plain and mate evals have to share one
    unit, because 3.5 squashes both through the same curve.
    """
    print("\neval conversion")

    #(label, mate flag, digits, expected sign)
    signs = (
        ("white mates in 3", "#", "3", +1),
        ("black mates in 3", "#", "-3", -1),
        ("white mates in 1", "#", "1", +1),
        ("black mates in 1", "#", "-1", -1),
        ("plain positive", "", "2.5", +1),
        ("plain negative", "", "-2.5", -1),
    )
    for label, flag, digits, want_sign in signs:
        ok, got = attempt(f"sign: {label}", eval_to_number, flag, digits)
        if ok:
            got_sign = (got > 0) - (got < 0)
            check(f"sign: {label}", got_sign == want_sign,
                  f"{flag}{digits} -> {got} (sign {got_sign}, want {want_sign})")

    ok, mate_one = attempt("mate in 1 beats mate in 5", eval_to_number, "#", "1")
    ok2, mate_five = attempt("mate in 1 beats mate in 5", eval_to_number, "#", "5")
    if ok and ok2:
        check("mate in 1 beats mate in 5", mate_one > mate_five,
              f"{mate_one} vs {mate_five}")

    ok, white_mate = attempt("white mate outranks black mate", eval_to_number, "#", "4")
    ok2, black_mate = attempt("white mate outranks black mate", eval_to_number, "#", "-4")
    if ok and ok2:
        check("white mate outranks black mate", white_mate > black_mate,
              f"#4 -> {white_mate}, #-4 -> {black_mate}")

    #Real games reach -24.05 and beyond, so test far past +9: in centipawns,
    #MATE has to sit above anything a plain eval can reach
    ok, crushing = attempt("mate outranks a merely winning score", eval_to_number, "", "50.0")
    if ok and ok2:
        check("mate outranks a merely winning score", white_mate > crushing,
              f"#4 -> {white_mate}, +50.0 -> {crushing}")

    #The mirror of "mate in 1 beats mate in 5", for the other side. From
    #White's POV a faster mate AGAINST White must be MORE negative, so the
    #comparison flips. Easy to get wrong with a sign applied to the wrong term.
    ok, black_one = attempt("black's mate in 1 beats its mate in 5",
                            eval_to_number, "#", "-1")
    ok2, black_five = attempt("black's mate in 1 beats its mate in 5",
                              eval_to_number, "#", "-5")
    if ok and ok2:
        check("black's mate in 1 beats its mate in 5", black_one < black_five,
              f"#-1 -> {black_one}, #-5 -> {black_five} "
              f"(the sooner mate must be the more negative)")

    #A mate score may approach the mate constant but must never pass it, or it
    #outranks mate itself and breaks any downstream mate-distance comparison
    for flag, digits in (("#", "1"), ("#", "3"), ("#", "-1"), ("#", "-3")):
        ok, value = attempt(f"|{flag}{digits}| stays inside MATE",
                            eval_to_number, flag, digits)
        if ok:
            check(f"|{flag}{digits}| stays inside MATE", abs(value) <= MATE,
                  f"{value} vs MATE {MATE}")

    #'#0' appears on a delivered mate. int('0') has no sign to read: the mate
    #belongs to whoever just moved, so the caller has to say who that was, and
    #a bare '#0' must raise rather than guess a side.
    for label, mover in (("#0 after white's move is +MATE", +1),
                         ("#0 after black's move is -MATE", -1)):
        ok, delivered = attempt(label, eval_to_number, "#", "0", mover=mover)
        if ok:
            check(label, delivered == mover * MATE,
                  f"#0 with mover {mover:+d} -> {delivered}")

    raised = False
    try:
        eval_to_number("#", "0")
    except Exception:
        raised = True
    check("bare #0 raises instead of guessing a side", raised)

    #Plain and mate evals must share a scale, or the squash in 3.5 flattens
    #every ordinary position to 0.5 while mate saturates
    def win_prob(value):
        return 1.0 / (1.0 + 10 ** (-value / 400.0))

    ok, one_pawn = attempt("a one-pawn edge is a meaningful win probability",
                           eval_to_number, "", "1.0")
    if ok:
        probability = win_prob(one_pawn)
        check("a one-pawn edge is a meaningful win probability",
              probability > 0.55,
              f"eval_to_number -> {one_pawn}, win_prob {probability:.4f} "
              f"(pawn units squash to ~0.50; centipawns to ~0.64)")


def test_eval_alignment():
    """
    The alignment contract: one entry per move PLAYED, in order. A move that
    carries no eval must still appear, or every position after it belongs to a
    different game and the board silently describes something that never
    happened.
    """
    print("\nevery played move survives tokenising, eval or not")
    cases = (
        ("gap in the middle",
         "1. e4 { [%eval 0.1] } 1... e5 2. Nf3 { [%eval 0.2] }",
         ["e4", "e5", "Nf3"]),
        ("clock-only comment",
         "1. e4 { [%eval 0.1] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%eval 0.2] }",
         ["e4", "e5", "Nf3"]),
        ("last move unannotated",
         "1. e4 { [%eval 0.1] } 1... e5 { [%eval 0.0] } 2. Nf3",
         ["e4", "e5", "Nf3"]),
        ("nothing annotated at all",
         "1. e4 e5 2. Nf3 Nc6",
         ["e4", "e5", "Nf3", "Nc6"]),
    )
    for name, text, want in cases:
        ok, pairs = attempt(f"alignment: {name}", san_text_prep_with_eval, text)
        if ok:
            got = moves_of(pairs)
            check(f"alignment: {name}", got == want, f"got {got!r}, want {want!r}")

    #The cases above only prove e5 survived. This proves what it survived AS:
    #a slot whose eval is None, which is what the training stage filters on
    ok, pairs = attempt("alignment: the gap comes back as None, not a dropped move",
                        san_text_prep_with_eval,
                        "1. e4 { [%eval 0.1] } 1... e5 2. Nf3 { [%eval 0.2] }")
    if ok:
        check("alignment: the gap comes back as None, not a dropped move",
              len(pairs) == 3 and pairs[1] == ("e5", None), f"got {pairs!r}")


# ---------------------------------------------------------------------------
# 2. Single moves, written out by hand
# ---------------------------------------------------------------------------

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

# (label, fen, san, expected origin, expected Move)
LITERAL = (
    # --- pawns -------------------------------------------------------------
    ("pawn single step",    START, "e3",  sq("e2"), Move(*sq("e3"))),
    ("pawn double step",    START, "e4",  sq("e2"), Move(*sq("e4"))),
    ("pawn, black",
     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -",
     "e5", sq("e7"), Move(*sq("e5"))),

    # --- pieces ------------------------------------------------------------
    ("knight from start",   START, "Nf3", sq("g1"), Move(*sq("f3"))),
    ("knight, queenside",   START, "Na3", sq("b1"), Move(*sq("a3"))),
    ("bishop",
     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
     "Bc4", sq("f1"), Move(*sq("c4"))),
    ("queen",
     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
     "Qh5", sq("d1"), Move(*sq("h5"))),
    ("king step",
     "8/8/8/8/8/8/8/4K2k w - -",
     "Kd1", sq("e1"), Move(*sq("d1"))),

    # --- captures ----------------------------------------------------------
    ("piece capture",
     "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -",
     "Nxe5", sq("f3"), Move(*sq("e5"))),
    ("pawn capture names its file",
     "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -",
     "exd5", sq("e4"), Move(*sq("d5"))),
    ("en passant looks like a pawn capture",
     "rnbqkbnr/ppppp1pp/8/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6",
     "exf6", sq("e5"), Move(*sq("f6"))),

    # --- promotions --------------------------------------------------------
    ("promotion to queen",  "8/P7/8/8/8/8/8/K6k w - -", "a8=Q", sq("a7"), Move(*sq("a8"), 5)),
    ("promotion to rook",   "8/P7/8/8/8/8/8/K6k w - -", "a8=R", sq("a7"), Move(*sq("a8"), 4)),
    ("promotion to bishop", "8/P7/8/8/8/8/8/K6k w - -", "a8=B", sq("a7"), Move(*sq("a8"), 3)),
    ("promotion to knight", "8/P7/8/8/8/8/8/K6k w - -", "a8=N", sq("a7"), Move(*sq("a8"), 2)),
    ("capture promotion",
     "1n6/P7/8/8/8/8/8/K6k w - -", "axb8=Q", sq("a7"), Move(*sq("b8"), 5)),

    # --- disambiguation ----------------------------------------------------
    # two knights, both able to reach d2: the file letter picks one
    ("disambiguate by file: Nbd2",
     "7k/8/8/8/8/5N2/8/1N2K3 w - -", "Nbd2", sq("b1"), Move(*sq("d2"))),
    ("disambiguate by file: Nfd2",
     "7k/8/8/8/8/5N2/8/1N2K3 w - -", "Nfd2", sq("f3"), Move(*sq("d2"))),
    # two rooks on the a-file, both able to reach a4: the rank digit picks one
    ("disambiguate by rank: R1a4",
     "R6k/8/8/8/8/8/8/R3K3 w - -", "R1a4", sq("a1"), Move(*sq("a4"))),
    ("disambiguate by rank: R8a4",
     "R6k/8/8/8/8/8/8/R3K3 w - -", "R8a4", sq("a8"), Move(*sq("a4"))),
    # three queens reach e5; file alone is ambiguous between a1 and a5
    ("disambiguate by both: Qa1e5",
     "8/8/7k/Q7/8/7K/8/Q3Q3 w - -", "Qa1e5", sq("a1"), Move(*sq("e5"))),
    ("disambiguate by both: Qa5e5",
     "8/8/7k/Q7/8/7K/8/Q3Q3 w - -", "Qa5e5", sq("a5"), Move(*sq("e5"))),

    # --- castling ----------------------------------------------------------
    ("white kingside",  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "O-O",   sq("e1"), Move(*sq("g1"))),
    ("white queenside", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "O-O-O", sq("e1"), Move(*sq("c1"))),
    ("black kingside",  "r3k2r/8/8/8/8/8/8/R3K2R b KQkq -", "O-O",   sq("e8"), Move(*sq("g8"))),
    ("black queenside", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq -", "O-O-O", sq("e8"), Move(*sq("c8"))),
)


def board_after(fen, origin, move):
    """The position reached by playing one move, as a FEN."""
    chess = from_fen(fen)
    if not make_move(chess, origin, move):
        return None
    return to_fen(chess)


def test_literal_moves():
    """
    Each case states the move the parser is supposed to find. Comparing the
    resulting FEN rather than the return value keeps this independent of
    whether san_make_move hands the move back.
    """
    print("\nsingle moves, expected square by square")
    for label, fen, san, origin, move in LITERAL:
        want = board_after(fen, origin, move)
        if want is None:
            check(f"{label}: TEST BUG, expected move is illegal", False,
                  f"{san} {origin}->{move}")
            continue

        chess = from_fen(fen)
        ok, _ = attempt(f"{label}: {san}", san_make_move, chess, san)
        if not ok:
            continue
        got = to_fen(chess)
        check(f"{label}: {san}", got == want,
              "" if got == want else f"landed on {got} not {want}")


# ---------------------------------------------------------------------------
# 3. Disambiguation has to respect legality, not just geometry
# ---------------------------------------------------------------------------

def test_pins_and_legality():
    """
    SAN omits a disambiguator when only one piece can LEGALLY make the move.
    A parser that picks by geometry alone chooses the pinned piece and then
    fails, or worse, succeeds with the wrong one.
    """
    print("\ndisambiguation respects pins")

    # Knights on c2 and e2 both reach d4 geometrically. The black rook on e8
    # pins the e2 knight against the king on e1, so only the c2 knight can
    # legally play it -- which is exactly why SAN writes a bare "Nd4" here.
    fen = "4r2k/8/8/8/8/8/2N1N3/4K3 w - -"
    chess = from_fen(fen)
    legal = [(o, m) for o, m in legal_move_list(chess)
             if make_move(chess, o, m) and (undo_move(chess) or True)]
    # Only one knight can legally reach d4 if the other is pinned
    reach_d4 = [(o, m) for o, m in legal
                if (m.row, m.col) == sq("d4")
                and abs(chess.chess_board[o[0]][o[1]]) == 2]
    check("test fixture: exactly one knight legally reaches d4",
          len(reach_d4) == 1, f"{len(reach_d4)} found")

    if len(reach_d4) == 1:
        origin, move = reach_d4[0]
        want = board_after(fen, origin, move)
        chess = from_fen(fen)
        ok, _ = attempt("undisambiguated Nd4 picks the unpinned knight",
                        san_make_move, chess, "Nd4")
        if ok:
            got = to_fen(chess)
            check("undisambiguated Nd4 picks the unpinned knight",
                  got == want, "" if got == want else f"got {got}")


# ---------------------------------------------------------------------------
# 4. Whole games, cross-checked against the UCI parser
# ---------------------------------------------------------------------------

#(name, san movetext, the same game in UCI)
GAMES = (
    ("scholar's mate",
     "1. e4 { [%eval 0.17] } 1... e5 { [%eval 0.19] } 2. Bc4 { [%eval 0.0] } "
     "2... Nc6 { [%eval 0.3] } 3. Qh5 { [%eval -0.5] } 3... Nf6 { [%eval 5.2] } "
     "4. Qxf7# { [%eval #0] } 1-0",
     "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7"),

    ("en passant",
     "1. e4 { [%eval 0.1] } 1... d5 { [%eval 0.2] } 2. e5 { [%eval 0.1] } "
     "2... f5 { [%eval 1.1] } 3. exf6 { [%eval 1.0] }",
     "e2e4 d7d5 e4e5 f7f5 e5f6"),

    ("both sides castle kingside",
     "1. e4 { [%eval 0.1] } 1... e5 { [%eval 0.2] } 2. Nf3 { [%eval 0.1] } "
     "2... Nf6 { [%eval 0.3] } 3. Bc4 { [%eval 0.2] } 3... Bc5 { [%eval 0.2] } "
     "4. O-O { [%eval 0.1] } 4... O-O { [%eval 0.2] }",
     "e2e4 e7e5 g1f3 g8f6 f1c4 f8c5 e1g1 e8g8"),

    ("queenside castling",
     "1. d4 { [%eval 0.1] } 1... d5 { [%eval 0.1] } 2. Nc3 { [%eval 0.0] } "
     "2... Nc6 { [%eval 0.1] } 3. Bf4 { [%eval 0.0] } 3... Bf5 { [%eval 0.1] } "
     "4. Qd2 { [%eval 0.0] } 4... Qd7 { [%eval 0.1] } 5. O-O-O { [%eval 0.0] } "
     "5... O-O-O { [%eval 0.1] }",
     "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8"),

    ("clocks alongside evals, the shape real movetext arrives in",
     "1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:03:00] } "
     "2. Nf3 { [%eval 0.15] [%clk 0:02:58] } 2... Nc6 { [%eval 0.2] [%clk 0:02:57] } 1-0",
     "e2e4 e7e5 g1f3 b8c6"),
)


def test_games_against_uci():
    """
    Replay the same game in both notations and compare after EVERY ply.
    Comparing only the final position lets two compensating bugs pass.
    """
    print("\nwhole games, ply by ply against the UCI parser")
    for name, san_text, uci_text in GAMES:
        reference = Chess()
        wanted = []
        for token in uci_text.split():
            origin, move = parse_uci_move(token)
            if not make_move(reference, origin, move):
                check(f"{name}: TEST BUG, uci move {token} is illegal", False)
                break
            wanted.append(to_fen(reference))
        else:
            ok, pairs = attempt(f"{name}: tokenises", san_text_prep_with_eval, san_text)
            if not ok:
                continue
            tokens = moves_of(pairs)
            if len(tokens) != len(wanted):
                check(f"{name}: token count matches the game", False,
                      f"{len(tokens)} tokens for {len(wanted)} moves: {tokens!r}")
                continue

            chess = Chess()
            drifted = None
            for index, (token, want) in enumerate(zip(tokens, wanted), 1):
                try:
                    san_make_move(chess, token)
                except Exception as error:
                    drifted = f"ply {index} ({token}) raised {type(error).__name__}: {error}"
                    break
                if to_fen(chess) != want:
                    drifted = (f"ply {index} ({token}) gave\n        {to_fen(chess)}"
                               f"\n   want {want}")
                    break
            check(f"{name}: {len(wanted)} plies match UCI", drifted is None,
                  drifted or "")


# ---------------------------------------------------------------------------
# 5. Bad input must fail loudly
# ---------------------------------------------------------------------------

def test_rejects_bad_input():
    """
    A parser that returns quietly on nonsense corrupts the dataset silently.
    Every one of these should raise.
    """
    print("\nmalformed input raises instead of corrupting the board")
    cases = (
        ("move no piece can make", START, "Nd5"),
        ("ambiguous without a disambiguator",
         "7k/8/8/8/8/5N2/8/1N2K3 w - -", "Nd2"),
        ("square that does not exist", START, "e9"),
        ("piece letter that does not exist", START, "Xe4"),
        ("promotion with no piece named", "8/P7/8/8/8/8/8/K6k w - -", "a8="),
        ("empty token", START, ""),
    )
    for label, fen, san in cases:
        chess = from_fen(fen)
        before = to_fen(chess)
        raised = False
        try:
            san_make_move(chess, san)
        except Exception:
            raised = True
        check(f"rejects {label}: {san!r}", raised,
              "" if raised else f"accepted it, board now {to_fen(chess)}")
        if not raised:
            check(f"  ...and left the board alone: {san!r}", to_fen(chess) == before)


# ---------------------------------------------------------------------------
# 6. Every legal move in a position round-trips
# ---------------------------------------------------------------------------

SUITE_FENS = (
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"),
    ("promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -"),
)


def test_every_legal_move_is_reachable():
    """
    Weaker than a full round trip, because there is no SAN writer to generate
    the strings. What it does prove: for every legal move, SOME san string the
    parser accepts produces exactly that move -- so no legal move is
    unreachable through the parser.

    Built from the move itself rather than from a generator, using the most
    explicit SAN form (always fully disambiguated), which every parser should
    accept even when shorter forms exist.
    """
    print("\nevery legal move is reachable through fully disambiguated SAN")
    letters = {1: "", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
    for name, fen in SUITE_FENS:
        chess = from_fen(fen)
        moves = []
        for origin, move in legal_move_list(chess):
            if make_move(chess, origin, move):
                undo_move(chess)
                moves.append((origin, move))

        bad = []
        for origin, move in moves:
            piece = abs(chess.chess_board[origin[0]][origin[1]])
            if piece == 6 and abs(move.col - origin[1]) == 2:
                continue            # castling has its own notation, tested above
            captures = chess.chess_board[move.row][move.col] != 0
            text = (letters[piece] + square_name(*origin)
                    + ("x" if captures else "") + square_name(move.row, move.col))
            if move.promotion:
                text += "=" + {2: "N", 3: "B", 4: "R", 5: "Q"}[move.promotion]

            want = board_after(fen, origin, move)
            trial = from_fen(fen)
            try:
                san_make_move(trial, text)
                if to_fen(trial) != want:
                    bad.append(f"{text} -> wrong square")
            except Exception as error:
                bad.append(f"{text} ({type(error).__name__})")

        check(f"{name}: {len(moves)} legal moves all reachable", not bad,
              "; ".join(bad[:4]) + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else ""))


# ---------------------------------------------------------------------------

def main():
    for test in (test_text_prep,
                 test_eval_conversion,
                 test_eval_alignment,
                 test_literal_moves,
                 test_pins_and_legality,
                 test_games_against_uci,
                 test_rejects_bad_input,
                 test_every_legal_move_is_reachable):
        try:
            test()
        except Exception:
            print(f"\n  {test.__name__} ABORTED")
            traceback.print_exc()
            FAIL.append(test.__name__)

    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())