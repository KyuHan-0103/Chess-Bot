import math
import time
from core.rules import *
from core.movegen import *
from engine.evaluate import self_evaluate
from engine.ordering import order_moves
from core.constants import MATE

#The deepest ply anything in this file will index or recurse to. Three
#separate things need the same bound, and they have to agree:
#   - killers is an array of this length, indexed by ply
#   - the killer update guards that index
#   - quiescence refuses to recurse past it
MAX_PLY = 64


class alpha_beta_engine:
    def __init__(self):
        self.t_table = Transposition_Table()
        #Replaced at the top of every search(). An unlimited budget is the
        #right default for anything calling get_best_move directly, which is
        #what the test suite does.
        self.limits = SearchLimits()

    #Everything the engine remembers between games. The arena calls this so a
    #new game never inherits scores describing positions from the last one.
    def new_game(self):
        self.t_table.clear()
        self.limits = SearchLimits()

    def negative_max(self, chess, depth, alpha, beta, ply):
        """
        This was an alpha beta search for the best move
        Now is a still an alpha beta search but seeks to maximize itself by minimizing the
        opponents move

        params:
        chess: the current chess board (also a chess object)
        depth: how many moves to look ahead (currently max for time efficiency is 5)
        alpha: the best score we have guaranteed higher up (initially -inf)
        beta: the best score the opponent has guaranteed for themselves higher up (initially inf)
        """
        #First, check time to see if we passed time or node limit
        self.limits.check()

        #A draw is worth 0 points. Don't include state evaluation with a draw score
        if chess.game is not None:
            return 0

        #Check if there is an entry for this current position
        #If a score exists, we want to use that score to either fail fast or find the best_move
        tt_score, tt_move = self.t_table.tt_probe(chess.zobrist, depth, alpha, beta, ply)
        if tt_score is not None and ply > 0:
            return tt_score
        
        #Because we cut at depth == 0 without checking for checks/mates, we at this guard
        #Extending here also means quiescence is never HANDED a position that is in check
        #from this side; the in-check branch down there is reached only by its own recursion.
        if ply < 30 and in_check(chess, chess.side_to_move):
            depth += 1
        if depth == 0:
            #ply is not incremented here because a move isn't made
            return self.quiescence(chess, alpha, beta, ply)
        
        #Get all legal moves in current state
        #order_moves wires in the tt_move to be first in line
        move_list = order_moves(chess, pseudo_legal_moves(chess), tt_move, self.t_table.killers,
                                self.t_table.history, ply)

        best_move, best_score = None, -math.inf
        legal_found = False
        original_alpha = alpha

        #Iterate through legal moves
        for origin, move in move_list:
                #Only recurse if move is legal
                if not make_move(chess, origin, move):
                    continue
                legal_found = True
                #Recurse down. The finally is what keeps the board consistent: if
                #check_time() raises Timeout anywhere below this frame, the exception
                #unwinds through here and the move still gets undone.
                try:
                    score = -self.negative_max(chess, depth - 1, -beta, -alpha, ply + 1)
                finally:
                    undo_move(chess)

                if score > best_score:
                    best_score = score
                    best_move = (origin, move)
                if best_score > alpha:
                    alpha = best_score
                #Prune Branch if the minimizer has a better option
                if alpha >= beta:
                    if ply < MAX_PLY and chess.chess_board[move.row][move.col] == 0 and move.promotion == 0:
                        self.remember_cutoff(origin, move, depth, ply)
                    break

        if not legal_found:  
            #Checks if no legal moves are found
            return -(MATE - ply) if in_check(chess, chess.side_to_move) else 0

        if best_score >= beta:
            flag = LOWER
        elif best_score <= original_alpha:
            flag = UPPER
        else:
            flag = EXACT

        #Store best move that could be made at this position
        self.t_table.tt_store(chess.zobrist, depth, best_score, flag, best_move, ply)
        return best_score
    
    def remember_cutoff(self, origin, move, depth, ply):
        killers = self.t_table.killers[ply]
        #Only shift if this is not already the most recent killer. Without
        #this the same move lands in both slots and the two-wide table
        #silently becomes one-wide.
        if killers[0] != (origin, move):
            self.t_table.killers[ply] = [(origin, move), killers[0]]
        #depth * depth: a cutoff at high remaining depth is backed by a whole
        #subtree of evidence and there are few such nodes, while shallow
        #cutoffs are nearly free and there are millions. Squaring stops the
        #shallow ones drowning the deep ones.
        self.t_table.history[origin[0] * 8 + origin[1]][move.row * 8 + move.col] += depth * depth

    def get_best_move(self, chess, depth, prev_best=None):
        #Returns the best move for player using Alpha-Beta search
        best_move, best_score = None, -math.inf
        #initialize bounds
        alpha, beta = -math.inf, math.inf

        hint = prev_best
        if hint is None:
            _, hint = self.t_table.tt_probe(chess.zobrist, depth, alpha, beta, 0)
        #Iterate through move list
        for origin, move in order_moves(chess, pseudo_legal_moves(chess), hint):
                #Only recurse if move is legal
                if not make_move(chess, origin, move):
                    continue
                try:
                    #The root is maximizing, so we want to minimize the opponents' score
                    score = -self.negative_max(chess, depth - 1, -beta, -alpha, 1)
                finally:
                    undo_move(chess)

                if score > best_score:
                    best_score, best_move = score, (origin, move)
                if best_score > alpha:
                    alpha = best_score

        #Save move to TT
        if best_move is not None:
            self.t_table.tt_store(chess.zobrist, depth, best_score, EXACT, best_move, 0)

        return best_move, best_score
    
    def quiescence(self, chess, alpha, beta, ply):
        #Check time and node limit
        self.limits.check()

        #Captures-only quiescence terminated on its own: every move it could
        #make consumed material, and material runs out. The in-check branch
        #generates quiet evasions, which consume nothing, so alternating
        #mutual checks could recurse forever. This is the backstop.
        if ply >= MAX_PLY:
            return self_evaluate(chess)

        if in_check(chess, chess.side_to_move):
            found_move = False
            for origin, move in order_moves(chess, pseudo_legal_moves(chess)):
                if not make_move(chess, origin, move):
                    continue
                found_move = True
                try:
                    score = -self.quiescence(chess, -beta, -alpha, ply + 1)
                finally:
                    undo_move(chess)

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

            #No legal move while in check is mate, and every evasion type is
            #generated above, so that conclusion is sound
            if found_move:
                return alpha
            return -(MATE - ply)

        #Not in check: stand pat, then captures and promotions only
        stand_still = self_evaluate(chess)
        if stand_still >= beta:
            return beta                         #Already too good, opponent avoids this
        if stand_still > alpha:
            alpha = stand_still                 #We don't have to take

        board = chess.chess_board
        captured = [(o, m) for o, m in pseudo_legal_moves(chess)
                    if board[m.row][m.col] != 0 or m.promotion]

        for origin, move in order_moves(chess, captured):
            if not make_move(chess, origin, move):
                continue
            try:
                score = -self.quiescence(chess, -beta, -alpha, ply + 1)
            finally:
                undo_move(chess)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def search(self, chess, max_depth, node_limit=None, time_limit=None):
        """
        Iterative deepening, one depth at a time until the budget runs out.

        Returns (best_move, best_score). best_move is None only when the
        position has no legal move at all -- never merely because the budget
        was tight. Callers read None as "game over", so the depth 1 guarantee
        below is what keeps that signal honest.
        """
        best_move, best_score = None, -math.inf
        #A fresh budget per search, held on the engine rather than the module
        self.limits = SearchLimits(node_limit, time_limit)
        #Making sure the board is properly returned to its initial state after the search
        key_before = chess.zobrist

        for depth in range(1, max_depth + 1):
            #Depth 1 runs unarmed so we are guaranteed to come back with a
            #legal move even on an absurdly short budget. Arming applies to
            #BOTH limits, which the old code got wrong: it suspended only the
            #deadline, so a node limit could still cut depth 1 short and leave
            #best_move as None.
            self.limits.armed = depth > 1

            try:
                move, score = self.get_best_move(chess, depth, best_move)
            except Timeout:
                break
            
            if move is not None:
                best_move, best_score = move, score
            #If mate is found, no need looking deeper
            if score > MATE - 1000:
                break

        assert chess.zobrist == key_before, "search left the board modified"

        return best_move, best_score
    #Nodes visited by the most recent search, for the arena's per-game totals
    def node_count(self):
        return self.limits.nodes

"""
NOTE FOR SELF: Why does this table need a 'flag'?
LOWER: It needs a flag because, if it cuts off early because score is >= beta, 
it isn't the true value, there are moves left to be searched. The true value is AT LEAST score.
UPPER: If nothing beat alpha, score <= original alpha, every move was checked with a narrow window,
the true value is at MOST best_score. (an upper bound)
EXACT: Gives a score we can actualy trust.

That is why we only trust a LOWER score if score is >= beta
or an UPPER score if score is <= alpha
"""
class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "move")
    def __init__(self, key, depth, score, flag, move):
        self.key, self.depth = key, depth
        self.score, self.flag, self.move = score, flag, move

EXACT, LOWER, UPPER = 0, 1, 2
TT_SIZE = 1 << 20

class Transposition_Table:
    def __init__(self):
        self.table = [None] * TT_SIZE
        self.killers = [[None, None] for _ in range(64)]
        self.history = [[0] * 64 for _ in range(64)]

    def tt_probe(self, key, depth, alpha, beta, ply):
        entry = self.table[key % TT_SIZE]
        if entry is None or entry.key != key:
            return None, None
        
        best_move = entry.move
        if entry.depth < depth:
            return None, best_move      #Too shallow to trust the score: move still needs to be explored

        #Unadjust mate distance
        score = entry.score
        if score > MATE - 1000:
            score -= ply
        elif score < -MATE + 1000:
            score += ply

        if entry.flag == EXACT:
            return score, best_move
        if entry.flag == LOWER and score >= beta:
            return score, best_move
        if entry.flag == UPPER and score <= alpha:
            return score, best_move
        return None, best_move

    def tt_store(self, key, depth, score, flag, move, ply):
        #Adjust mate score
        if score > MATE - 1000:
            score += ply
        elif score < -MATE + 1000:
            score -= ply

        self.table[key % TT_SIZE] = TTEntry(key, depth, score, flag, move)

    def clear(self):
        """Forget every entry. Called between games, never between searches."""
        self.table = [None] * TT_SIZE
        self.killers = [[None, None] for _ in range(64)]
        self.history = [[0] * 64 for _ in range(64)]
        
class Timeout(Exception):
    """Raised from deep inside the tree when the budget for this search runs out."""
    pass

class SearchLimits:
    """
    The budget one search may spend, and the counter it spends against.

    Two limits live here, and they are deliberately separate numbers:

        node_limit  a hard ceiling on nodes visited. None means unlimited.
        deadline    an absolute perf_counter() time to stop by. None means
                    unlimited.

    POLL_INTERVAL is neither of them. Reading the clock at every node would
    cost more than the search does, so the clock is only read every
    POLL_INTERVAL nodes. It is a sampling rate, not a budget, which is why it
    is a class constant rather than something a caller passes in. 
    """

    #Nodes between reads of the wall clock
    POLL_INTERVAL = 2048

    def __init__(self, node_limit=None, time_limit=None):
        self.node_limit = node_limit
        self.deadline = time.perf_counter() + time_limit if time_limit else None
        self.nodes = 0
        #While False, check() still counts nodes but never raises. Depth 1 runs
        #unarmed so a search always comes back with a legal move, however tight
        #the budget it was given.
        self.armed = True

    def check(self):
        """Count this node, and stop the search if the budget is spent."""
        self.nodes += 1
        if not self.armed:
            return
        if self.node_limit is not None and self.nodes >= self.node_limit:
            raise Timeout
        #Counting first means the first clock read lands on POLL_INTERVAL, not
        #on node zero where the deadline can never have passed yet
        if self.deadline is not None and self.nodes % self.POLL_INTERVAL == 0:
            if time.perf_counter() > self.deadline:
                raise Timeout