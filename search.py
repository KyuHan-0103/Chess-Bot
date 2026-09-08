import math
import time
from rules import *
from movegen import *
from evaluate import self_evaluate
from ordering import order_moves
from constants import MATE

def negative_max(chess, depth, alpha, beta, ply):
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
    #First, check time to see if we passed time limit
    check_time()

    #A draw is worth 0 points. Don't include state evaluation with a draw score
    if chess.game is not None:
        return 0

    #Check if there is an entry for this current position
    #If a score exists, we want to use that score to either fail fast or find the best_move
    tt_score, tt_move = tt_probe(chess.zobrist, depth, alpha, beta, ply)
    if tt_score is not None and ply > 0:
        return tt_score
    
    #Because we cut at depth == 0 without checking for checks/mates, we at this guard
    #Note ply is capped at 20 so it doesn't get stuck
    #NOTE: a more thorough check would be if the side to move is in check, we put all possible evasions
    #in the quiescence function and evaluate all of them
    if ply < 30 and in_check(chess, chess.side_to_move):
        depth += 1
    if depth == 0:
        return quiescence(chess, alpha, beta)
    
    #Get all legal moves in current state
    #order_moves wires in the tt_move to be first in line
    move_list = order_moves(chess, pseudo_legal_moves(chess), tt_move)

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
                score = -negative_max(chess, depth - 1, -beta, -alpha, ply + 1)
            finally:
                undo_move(chess)

            if score > best_score:
                best_score = score
                best_move = (origin, move)
            if best_score > alpha:
                alpha = best_score
            #Prune Branch if the minimizer has a better option
            if alpha >= beta:
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
    tt_store(chess.zobrist, depth, best_score, flag, best_move, ply)
    return best_score

def get_best_move(chess, depth, prev_best=None):
    #Returns the best move for player using Alpha-Beta search
    best_move, best_score = None, -math.inf
    #initialize bounds
    alpha, beta = -math.inf, math.inf

    hint = prev_best
    if hint is None:
        _, hint = tt_probe(chess.zobrist, depth, alpha, beta, 0)
    #Iterate through move list
    for origin, move in order_moves(chess, pseudo_legal_moves(chess), hint):
            #Only recurse if move is legal
            if not make_move(chess, origin, move):
                continue

            try:
                #The root is maximizing, so we want to minimize the opponents' score
                score = -negative_max(chess, depth - 1, -beta, -alpha, 1)
            finally:
                undo_move(chess)

            if score > best_score:
                best_score, best_move = score, (origin, move)
            if best_score > alpha:
                alpha = best_score

    #Save move to TT
    if best_move is not None:
        tt_store(chess.zobrist, depth, best_score, EXACT, best_move, 0)

    return best_move, best_score

def quiescence(chess, alpha, beta):

    #Check time limit
    check_time()

    stand_still = self_evaluate(chess)
    if stand_still >= beta:
        return beta                         #Already too good, opponent avoids this
    if stand_still > alpha:
        alpha = stand_still                  #We don't have to take

    """
    To Implement:
    Check if in check. If so evaluate all evasions. If no evasions, return MATE value
    """

    board = chess.chess_board
    captured = [(o, m) for o, m in pseudo_legal_moves(chess)
                if board[m.row][m.col] != 0 or m.promotion]

    for origin, move in order_moves(chess, captured):
        if not make_move(chess, origin, move):
            continue
        try:
            score = -quiescence(chess, -beta, -alpha)
        finally:
            undo_move(chess)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def search(chess, max_depth, time_limit=None):
    global _deadline, _nodes
    best_move, best_score = None, -math.inf
    limit = time.perf_counter() + time_limit if time_limit else None
    _nodes = 0

    #Making sure the board is properly returned to its initial state after the search
    key_before = chess.zobrist
    try:
        for depth in range(1, max_depth + 1):
            #Depth 1 runs with no deadline so we are guaranteed to come back
            #with a legal move even on an absurdly short time limit
            _deadline = None if depth == 1 else limit

            try:
                move, score = get_best_move(chess, depth, best_move)
            except Timeout:
                break
            
            if move is not None:
                best_move, best_score = move, score
            #If mate is found, no need looking deeper
            if score > MATE - 1000:
                break
    finally:
        _deadline = None

    assert chess.zobrist == key_before, "search left the board modified"

    return best_move, best_score

class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "move")
    def __init__(self, key, depth, score, flag, move):
        self.key, self.depth = key, depth
        self.score, self.flag, self.move = score, flag, move

EXACT, LOWER, UPPER = 0, 1, 2
TT_SIZE = 1 << 20
table = [None] * TT_SIZE

def tt_probe(key, depth, alpha, beta, ply):
    entry = table[key % TT_SIZE]
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

def tt_store(key, depth, score, flag, move, ply):
    #Adjust mate score
    if score > MATE - 1000:
        score += ply
    elif score < -MATE + 1000:
        score -= ply

    table[key % TT_SIZE] = TTEntry(key, depth, score, flag, move)

class Timeout(Exception):
    pass
_deadline = None
_nodes = 0
def check_time():
    global _nodes
    _nodes += 1
    if _deadline and _nodes % 2048 == 0 and time.perf_counter() > _deadline:
        raise Timeout