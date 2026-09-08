from core.constants import *

#Attack detection along one ray: the first occupied square only, or 0 if the
#ray runs off the board. Deliberately separate from line_moves, which has the
#opposite contract.
def first_blocker(chess, row, col, row_step, col_step):
    board = chess.chess_board
    new_row = row + row_step
    new_col = col + col_step
 
    while on_board(new_row, new_col):
        if board[new_row][new_col] != 0:
            return board[new_row][new_col]
        new_row += row_step
        new_col += col_step
 
    return 0

#Returns true if this square can be attacked by an enemy piece
def is_guarded(chess, row, col, side) -> bool:
    board = chess.chess_board
    enemy = -side

    #Knights
    for row_add, col_add in KNIGHT_MOVES:
        knight_row = row + row_add
        knight_col = col + col_add
        if on_board(knight_row, knight_col) and board[knight_row][knight_col] == (KNIGHT * enemy):
            return True

    #King (only the 8 squares around the king)
    for row_add, col_add in KING_MOVES:
        new_row = row + row_add
        new_col = col + col_add
        if on_board(new_row, new_col) and board[new_row][new_col] == KING * enemy:
            return True

    #For pawns. An enemy pawn attacks diagonally and 1 step in the forward direction
    pawn_row = row + forward(side)
    for col_add in (-1, 1):
        if on_board(pawn_row, col + col_add) and board[pawn_row][col + col_add] == PAWN * enemy:
            return True

    #For Rooks and Queens
    for row_add, col_add in ORTHOGONAL:
        blocker = first_blocker(chess, row, col, row_add, col_add)
        if blocker == ROOK * enemy or blocker == QUEEN * enemy:
            return True

    #For Bishops and Queens
    for row_add, col_add in DIAGONAL:
        blocker = first_blocker(chess, row, col, row_add, col_add)
        if blocker == BISHOP * enemy or blocker == QUEEN * enemy:
            return True

    #PASSED all tests
    return False

#Returns true if the given side's king currently attacked
def in_check(chess, side):
    king_row, king_col = chess.white_king_pos if side > 0 else chess.black_king_pos
    return is_guarded(chess, king_row, king_col, side)