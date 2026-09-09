import argparse
import math
import sys
import time
 
from core.board import Chess
from core.notation import from_fen, to_fen, move_name
from core.rules import make_move, game_result
from engine.player import RandomPlayer, SearchPlayer
 
#Two engines from the same start position play the same game every time: the
#zobrist seed is fixed and the search is deterministic. A 100 game match would
#be one game repeated 100 times, so the openings are what create the variety.
#Kept short and roughly balanced on purpose; move them to a file once there
#are more than a screenful.
DEFAULT_OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppppp1p/6p1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
    "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
]
 
#Outcomes are always stored from WHITE's point of view, because that is the
#only perspective that does not change halfway through a colour-swapped match
WHITE_WIN, DRAW, BLACK_WIN = 1.0, 0.5, 0.0

class Adjudicator:
    """
    Decides when to stop a game early, and why.
 
    core.rules already ends games properly: checkmate, stalemate, threefold
    repetition, the fifty move rule and insufficient material. What it cannot
    do is end a game that is *obviously* over but still legal, and those are
    the games that eat a match's compute budget.
 
    Three rules, in decreasing order of how much time they save:
 
        draw        Both players have reported a near-zero score for a while.
                    BOTH matters: one engine's evaluation can be wrong, two
                    independently agreeing rarely are.
        resign      One player has reported a hopeless score for a while. 
        ply_limit   A safety net so no game can run forever.
 
    Scores arrive from each player's OWN point of view, which is what
    self_evaluate returns, so "losing badly" is always a large negative number
    whichever colour reported it.
    """
    def __init__(self, ply_limit=350, draw_margin=15, draw_moves=10,
            draw_after_move=30, resign_margin=900, resign_moves=6):
        self.ply_limit = ply_limit
        self.draw_margin = draw_margin
        self.draw_moves = draw_moves
        self.draw_after_move = draw_after_move
        self.resign_margin = resign_margin
        self.resign_moves = resign_moves
        self.reset()

    def reset(self):
        #Consecutive plies in which the mover called the position level
        self.level_run = 0
        #Consecutive plies in which each side called its own position hopeless
        self.losing_run = {1: 0, -1: 0}

    #Called after every move, with the score the side that just moved
    #reported. A player with no opinion passes None and is not counted.
    def record(self, side, score):
        if score is None:
            self.level_run = 0
            self.losing_run[side] = 0
            return
 
        if abs(score) <= self.draw_margin:
            self.level_run += 1
        else:
            self.level_run = 0
 
        if score <= -self.resign_margin:
            self.losing_run[side] += 1
        else:
            self.losing_run[side] = 0
 
    #None while the game should continue, otherwise (outcome, reason).
    #ply is the number of plies played so far.
    def verdict(self, ply):
        #Both sides have to have agreed, hence the doubling: draw_moves full
        #moves is 2 * draw_moves plies
        if (ply >= 2 * self.draw_after_move
                and self.level_run >= 2 * self.draw_moves):
            return DRAW, "Adjudicated draw"
 
        for side in (1, -1):
            if self.losing_run[side] >= self.resign_moves:
                loser_is_white = side > 0
                return (BLACK_WIN if loser_is_white else WHITE_WIN), "Adjudicated win"
 
        if ply >= self.ply_limit:
            return DRAW, "Ply limit"
        return None

class GameResult:
    """
    Everything worth knowing about one finished game.
 
    A bare "white won" is useless the moment something looks wrong. The start
    FEN plus the move list is a complete, replayable, diffable record, which is
    what turns a loss into a regression test.
    """
 
    __slots__ = ("start_fen", "moves", "outcome", "reason", "plies",
                 "white_name", "black_name", "white_nodes", "black_nodes",
                 "error", "seconds")
 
    def __init__(self, start_fen, white_name, black_name):
        self.start_fen = start_fen
        self.white_name = white_name
        self.black_name = black_name
        self.moves = []
        self.outcome = None
        self.reason = None
        self.plies = 0
        self.white_nodes = 0
        self.black_nodes = 0
        self.error = None
        self.seconds = 0.0
 
    #Score from a given player's point of view, so a colour-swapped match can
    #be totalled without the caller doing the flipping by hand
    def score_for(self, name):
        if self.outcome is None:
            return None
        if name == self.white_name:
            return self.outcome
        return 1.0 - self.outcome
 
    def line(self):
        """One line summary, for watching a match go by."""
        text = (f"{self.white_name} vs {self.black_name}: "
                f"{self.outcome if self.outcome is not None else '?'} "
                f"({self.reason}, {self.plies} plies, {self.seconds:.1f}s)")
        return text + (f"  ERROR: {self.error}" if self.error else "")
 
    def record(self):
        """The replayable form: a start position and the moves played from it."""
        return f'{self.start_fen}\nmoves {" ".join(self.moves)}'
 
 
def play_one_game(white, black, start_fen=None, adjudicator=None):
    """
    Play one game and return a GameResult.
 
    Players are passed in, never constructed here, so their state survives a
    whole match and is cleared deliberately by new_game() below.
    """
    chess = from_fen(start_fen) if start_fen else Chess()
    start_fen = to_fen(chess)
    adjudicator = adjudicator or Adjudicator()
    adjudicator.reset()
 
    white.new_game()
    black.new_game()
 
    result = GameResult(start_fen, getattr(white, "name", "white"),
                        getattr(black, "name", "black"))
    started = time.perf_counter()
 
    while True:
        #Termination is judged BEFORE anyone is asked to think, so a player is
        #never handed a position whose game is already over
        ended = game_result(chess)
        if ended is not None:
            if ended == "Checkmate":
                #side_to_move is the side that has been mated
                result.outcome = BLACK_WIN if chess.side_to_move > 0 else WHITE_WIN
            else:
                result.outcome = DRAW
            result.reason = ended
            break
 
        early = adjudicator.verdict(result.plies)
        if early is not None:
            result.outcome, result.reason = early
            break
 
        side = chess.side_to_move
        player = white if side > 0 else black
 
        who = getattr(player, "name", "?")
        #A player is allowed to be buggy without taking the whole match down
        #with it. Anything it raises is caught, recorded against the position
        #it happened in, and this one game is abandoned. make_move raises
        #rather than returning False for a move from an empty square, so
        #catching only the False case is not enough.
        position = to_fen(chess)
        try:
            move = player.select_move(chess)
        except Exception as error:
            result.error = f"{who} raised {type(error).__name__}: {error} in {position}"
            result.reason = "Player error"
            break
 
        if move is None:
            #game_result above already proved a legal move exists, so this is
            #the player's bug and not the end of the game
            result.error = f"{who} returned None with legal moves available in {position}"
            result.reason = "Player error"
            break
 
        origin, target = move
        try:
            legal = make_move(chess, origin, target)
        except Exception as error:
            result.error = (f"{who} played unplayable move {move} -> "
                            f"{type(error).__name__}: {error} in {position}")
            result.reason = "Illegal move"
            break
        if not legal:
            result.error = (f"{who} played illegal move "
                            f"{move_name(origin, target)} in {position}")
            result.reason = "Illegal move"
            break
 
        result.moves.append(move_name(origin, target))
        result.plies += 1
        if side > 0:
            result.white_nodes += player.last_nodes()
        else:
            result.black_nodes += player.last_nodes()
 
        adjudicator.record(side, player.last_score())
 
    result.seconds = time.perf_counter() - started
    return result
 
 
def play_match(player_a, player_b, openings=None, games=20, adjudicator=None):
    """
    Play `games` games and yield each GameResult as it finishes.
 
    A generator rather than a list, so a match that runs for hours can report
    progress, and so an early stopping rule can simply stop consuming it.
 
    Every opening is played twice with the colours reversed. White's advantage
    is real and large, so without the swap part of what gets measured is who
    drew white.
    """
    openings = openings or DEFAULT_OPENINGS
    adjudicator = adjudicator or Adjudicator()
 
    for index in range(games):
        #Two consecutive games share an opening and swap colours
        opening = openings[(index // 2) % len(openings)]
        a_is_white = index % 2 == 0
        white, black = (player_a, player_b) if a_is_white else (player_b, player_a)
        yield play_one_game(white, black, opening, adjudicator)
 
 
"""
===========================================================================
Result aggregation
===========================================================================
Pure functions over finished games. Nothing accumulates during the match, so
a saved list of GameResults can be re-summarised, sliced by opening or fed to
a different test without replaying a single game.
"""
 
 
def elo_from_score(score):
    """
    Score rate (0..1) to an Elo difference.
 
    The inverse of the logistic Elo curve. Undefined at the ends: a clean
    sweep is evidence of "at least this much stronger", not of infinity, so
    the caller gets an infinity it has to render sensibly.
    """
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score - 1.0)
 
 
def aggregate(results, name_a, name_b):
    """
    Turn finished games into a verdict.
 
    The Elo point estimate on its own is a trap: 55-45 over 100 games is well
    inside noise for two equal engines. The interval is the part that says
    whether anything was learned.
    """
    scored = [r for r in results if r.outcome is not None and r.error is None]
    errors = [r for r in results if r.error is not None]
 
    points = [r.score_for(name_a) for r in scored]
    n = len(points)
 
    summary = {
        "games": n,
        "errors": errors,
        "wins": sum(1 for p in points if p == 1.0),
        "draws": sum(1 for p in points if p == 0.5),
        "losses": sum(1 for p in points if p == 0.0),
        "reasons": {},
        "as_white": [], "as_black": [],
        "plies": [r.plies for r in scored],
        "nodes_a": 0, "nodes_b": 0,
    }
 
    for r in scored:
        summary["reasons"][r.reason] = summary["reasons"].get(r.reason, 0) + 1
        if r.white_name == name_a:
            summary["as_white"].append(r.outcome)
            summary["nodes_a"] += r.white_nodes
            summary["nodes_b"] += r.black_nodes
        else:
            summary["as_black"].append(1.0 - r.outcome)
            summary["nodes_a"] += r.black_nodes
            summary["nodes_b"] += r.white_nodes
 
    if n == 0:
        summary["score"] = None
        return summary
 
    mean = sum(points) / n
    summary["score"] = mean
    summary["elo"] = elo_from_score(mean)
 
    #Standard error of the mean of the per-game scores. Using the observed
    #spread rather than a formula keeps draws weighted correctly: a match of
    #all draws has no variance and so no uncertainty about being level.
    if n > 1:
        variance = sum((p - mean) ** 2 for p in points) / (n - 1)
        error = math.sqrt(variance / n)
    else:
        error = 0.0
    #95% interval, clamped because a score rate cannot leave [0, 1]
    low = min(max(mean - 1.96 * error, 0.0), 1.0)
    high = min(max(mean + 1.96 * error, 0.0), 1.0)
    summary["score_interval"] = (low, high)
    summary["elo_interval"] = (elo_from_score(low), elo_from_score(high))
    return summary
 
 
def format_elo(value):
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:+.0f}"
 
 
def print_report(summary, name_a, name_b):
    print("\n" + "=" * 66)
    print(f"{name_a}  vs  {name_b}")
    print("=" * 66)
 
    if not summary["games"]:
        print("no completed games")
    else:
        print(f"  games      {summary['games']}   "
              f"+{summary['wins']} ={summary['draws']} -{summary['losses']}")
        print(f"  score      {summary['score']:.3f}  "
              f"[{summary['score_interval'][0]:.3f}, {summary['score_interval'][1]:.3f}]")
        low, high = summary["elo_interval"]
        print(f"  elo        {format_elo(summary['elo'])}  "
              f"[{format_elo(low)}, {format_elo(high)}]")
 
        #A large gap here between two engines that should be symmetric is the
        #fastest way to spot a sign error in the evaluation
        for label, points in (("as white", summary["as_white"]),
                              ("as black", summary["as_black"])):
            if points:
                print(f"  {label}   {sum(points) / len(points):.3f}  ({len(points)} games)")
 
        plies = summary["plies"]
        print(f"  length     {sum(plies) / len(plies):.0f} plies average, "
              f"{max(plies)} longest")
        print(f"  nodes      {name_a}: {summary['nodes_a']:,}   "
              f"{name_b}: {summary['nodes_b']:,}")
 
        #The harness's own sanity check. Mostly "Ply limit" means the
        #adjudicator is mistuned and the Elo number above is not trustworthy.
        print("  endings")
        for reason, count in sorted(summary["reasons"].items(),
                                    key=lambda kv: -kv[1]):
            print(f"    {count:>4}  {reason}")
 
    if summary["errors"]:
        print(f"\n  {len(summary['errors'])} ERRORS (bugs, not results):")
        for r in summary["errors"][:5]:
            print(f"    {r.error}")
            print(f"      {r.record()}")
    print("=" * 66)
 
 
def main():
    parser = argparse.ArgumentParser(description="Play a match between two players")
    parser.add_argument("--games", type=int, default=10,
                        help="games to play, rounded to a colour-swapped pair (default 10)")
    parser.add_argument("--depth", type=int, default=6, help="max search depth (default 6)")
    parser.add_argument("--nodes", type=int, default=20000,
                        help="node budget per move; reproducible, unlike a time budget")
    parser.add_argument("--time", type=float, default=None,
                        help="seconds per move instead of a node budget")
    parser.add_argument("--random", action="store_true",
                        help="play the engine against a random mover as a smoke test")
    parser.add_argument("--ply-limit", type=int, default=350)
    parser.add_argument("--pgn", help="file to append game records to")
    args = parser.parse_args()
 
    player_a = SearchPlayer(args.depth, args.nodes, args.time, name="A")
    if args.random:
        player_b = RandomPlayer(seed=0, name="B(random)")
    else:
        player_b = SearchPlayer(args.depth, args.nodes, args.time, name="B")
 
    adjudicator = Adjudicator(ply_limit=args.ply_limit)
    print(f"{player_a.name} vs {player_b.name}, {args.games} games")
 
    results = []
    for index, result in enumerate(play_match(player_a, player_b,
                                              games=args.games,
                                              adjudicator=adjudicator), 1):
        results.append(result)
        print(f"  [{index:>3}/{args.games}] {result.line()}")
        if args.pgn:
            with open(args.pgn, "a") as handle:
                handle.write(result.record() + "\n\n")
 
    summary = aggregate(results, player_a.name, player_b.name)
    print_report(summary, player_a.name, player_b.name)
    #A non-zero exit for errors, so this can sit in a script
    return 1 if summary["errors"] else 0
 
 
if __name__ == "__main__":
    sys.exit(main()) 