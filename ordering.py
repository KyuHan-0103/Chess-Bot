from operator import itemgetter
from constants import PIECE_VALUES
from movegen import same_move

#One thing to note, this function orders moves by capture score, but it doesn't see en_passant captures
def order_moves(chess, moves, tt_move=None):
    board = chess.chess_board
    scored = []

    for origin, move in moves:
        #If we have a move from our transposition table or a previous best move,
        #We want to check that move first
        if tt_move is not None and same_move((origin, move), tt_move):
            scored.append((10_000_000, origin, move))
            continue

        victim = board[move.row][move.col]
        if victim:
            attacker = board[origin[0]][origin[1]]
            score = 10 * PIECE_VALUES[abs(victim)] - PIECE_VALUES[abs(attacker)]
        else: score = 0

        if move.promotion != 0:
            score += PIECE_VALUES[abs(move.promotion)]
        scored.append((score, origin, move))
    
    scored.sort(key=itemgetter(0), reverse=True)
    return [(origin, move) for _, origin, move in scored]

killers = [[None, None] for _ in range(64)]
history = [[0] * 64 for _ in range(64)]