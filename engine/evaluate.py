from core.constants import *
from engine.psqt import *

def self_evaluate(chess):
    """
    Material plus placement, from the side-to-move POV.

    No PIECE_VALUES here: TABLE already has the material folded in, so adding
    them again would count every piece twice.

    The two totals stay separate integers through the loop and blend once on
    the way out. A weighted sum is linear:

        sum(w * mid + (1 - w) * end)  ==  w * sum(mid) + (1 - w) * sum(end)

    so this is two float multiplies for the whole board instead of two per
    piece, and the loop itself never touches a float.
    """
    board = chess.chess_board
    mid_table = TABLE[MIDGAME]
    end_table = TABLE[ENDGAME]

    middle = end = 0

    #The table already knows which row a black piece reads, so both loops look
    #the same. Only the sign differs: my pieces count for me, yours against me.
    for row, col in chess.white_pieces:
        piece = board[row][col]
        middle += mid_table[piece][row][col]
        end += end_table[piece][row][col]

    for row, col in chess.black_pieces:
        piece = board[row][col]
        middle -= mid_table[piece][row][col]
        end -= end_table[piece][row][col]
            
    #Clamped because a promotion can put more than the starting material on the
    #board. Past 1.0 the endgame weight goes negative and we extrapolate past
    #the midgame table instead of interpolating between the two.
    middle_game = min(chess.phase, MAX_PHASE) / MAX_PHASE
    score = middle * middle_game + end * (1 - middle_game)

    #Everything above is from white's point of view
    return score * chess.side_to_move

#Converts board to a list of indices for my network to make use of
def indexer(chess):
    board = chess.chess_board
    res = []
    for white_piece, black_piece in zip(chess.white_pieces, chess.black_pieces):
        white_square = (white_piece[0] * 8) + white_piece[1]
        black_square = (black_square[0] * 8) + black_piece[1]
        res.append((board[white_piece[0]][white_piece[1]] + 6) * 64 + white_square)
        res.append((board[black_piece[0]][black_piece[1]] + 6) * 64 + black_square)
