from typing import NamedTuple
from core.constants import *
from core.attacks import *

#Creating a move class to differentiate promotion moves by the piece they promote to
class Move(NamedTuple):
    row: int
    col: int
    promotion: int = 0 


#Gets all available moves (both legal and non legal) for the piece at position (row, col)
def available_moves(chess, row, col):
    piece = chess.chess_board[row][col]
    if piece == 0:
        return []
    
    piece_type = abs(piece)
    if piece_type == KING:
        return king_moves(chess, row, col)
    if piece_type == QUEEN:
        return queen_moves(chess, row, col)
    if piece_type == ROOK:
        return rook_moves(chess, row, col)
    if piece_type == BISHOP:
        return bishop_moves(chess, row, col)
    if piece_type == KNIGHT:
        return knight_moves(chess, row, col)
    if piece_type == PAWN:
        return pawn_moves(chess, row, col)

"""Line based moves"""
#Gets all moves a piece can make incrementing for each element of direction
#Keeps going until off the board or meets a blocker. If blocker is an enemy
#include that square
def line_moves(chess, row, col, directions):
    #Initiate board, side, and move list
    board = chess.chess_board
    side = side_of(board[row][col])
    move_list = []

    #For each tuple of directions, keep stepping until meeting a blocker
    #If blocker is enemy include that blocker in move list else break
    for r, c in directions:
        new_row = row + r
        new_col = col + c
        while on_board(new_row, new_col):
            blocker = board[new_row][new_col]
            #If no piece occupies that square add it
            if blocker == 0:
                move_list.append(Move(new_row, new_col))
            else:
                #If the occupying piece is an enemy add it and break
                if is_enemy(blocker, side):
                    move_list.append(Move(new_row, new_col))
                break
            #Increment row and column in while loop
            new_row += r
            new_col += c

    return move_list

#Gets all moves for a queen at the position (row, col)
def queen_moves(chess, row, col):
    return rook_moves(chess, row, col) + bishop_moves(chess, row, col)

#Gets all moves for a rook at the position (row, col)
def rook_moves(chess, row, col):
    return line_moves(chess, row, col, ORTHOGONAL)

#Gets all moves for a bishop at the position (row, col)
def bishop_moves(chess, row, col):
    return line_moves(chess, row, col, DIAGONAL)

"""Knight moves"""
#Gets all moves for a knight at the position (row, col)
def knight_moves(chess, row, col):
    #Gets board and knight piece
    board = chess.chess_board
    knight = board[row][col]

    #Gets side of the knight
    side = side_of(knight)

    #Create list of tuples to hold the moves
    move_list = []

    for row_add, col_add in KNIGHT_MOVES:
        new_row = row + row_add
        new_col = col + col_add
        if on_board(new_row, new_col) and not is_friendly(board[new_row][new_col], side):
            move_list.append(Move(new_row, new_col))

    return move_list

"""Stepping pieces"""
#Gets all moves for a king at the position (row, col)
def king_moves(chess, row, col):
    #Gets board and king piece
    board = chess.chess_board
    king = board[row][col]

    #Gets side of the king
    side = side_of(king)

    #Create list of tuples to hold the moves
    move_list = []

    for row_add, col_add in KING_MOVES:
        #New coords of potential moves
        new_row = row + row_add
        new_col = col + col_add
        #Checks if new coords are inbounds
        if not on_board(new_row, new_col):
            continue
        #Cannot land on your own pieces
        if is_friendly(board[new_row][new_col], side):
            continue
        #Makes sure the square isn't guarded
        if not is_guarded(chess, new_row, new_col, side):
            move_list.append(Move(new_row, new_col))
    
    #Checks if king can castle
    move_list += can_castle(chess, row, col, side)

    return move_list

#Gets all moves for a pawn at the position (row, col)
def pawn_moves(chess, row, col):
    #Gets board and pawn piece
    board = chess.chess_board
    pawn = board[row][col]

    #Gets side of the pawn
    side = side_of(pawn)
    step = forward(side)
    #Create list of tuples to hold the moves
    move_list = []

    #Checks if the pawn can move up one sqaure (if there is anything obstructing it)
    one_row = row + step
    if on_board(one_row, col) and board[one_row][col] == 0  :
        #Checks for promotions (going forward)
        if one_row == promotion_row(side):
            for piece_type in PROMOTION_PIECES:
                move_list.append(Move(one_row, col, piece_type))
        else:
            #If not a promotion, simply append move forward
            move_list.append(Move(one_row, col))
        #Checks if the pawn go move up two squares (if there are any obstructions and if it is on its starting square)
        two_row = one_row + step
        if on_board(two_row, col) and row == starting_row(side) and board[two_row][col] == 0:
            move_list.append(Move(two_row, col))

    #Checks if pawn can eat diagonally
    for col_add in (-1, 1):
        new_col = col_add + col
        if on_board(one_row, new_col) and is_enemy(board[one_row][new_col], side):
            if one_row == promotion_row(side):
                for piece_type in PROMOTION_PIECES:
                    move_list.append(Move(one_row, new_col, piece_type))
            else:
                move_list.append(Move(one_row, new_col))

    #Checks En Passant
    if chess.en_passant.get((row, col)) is not None:
        move_list.append((chess.en_passant[(row, col)]))
    
    return move_list

"""Castling functions"""
#Returns castling moves if the king on the passed down side can castle
def can_castle(chess, row, col, side):
    castle_list = get_castling_rights(chess, side)
    if len(castle_list) == 0:
        return []

    board = chess.chess_board
    move_list = []

    #Cannot castle out of check
    if is_guarded(chess, row, col, side):
        return []
    
    for side_dir in castle_list:
        if side_dir == 6:
            rook_col, col_step, king_dest = 7, 1, 6
        else:
            rook_col, col_step, king_dest = 0, -1, 2

        #Every square between king and rook must be empty
        blocked = False
        scan = col + col_step
        while scan != rook_col:
            if board[row][scan] != 0:
                blocked = True
                break
            scan += col_step
        if blocked:
            continue

        #The king may not pass through or land on an attacked square
        if is_guarded(chess, row, col + col_step, side):
            continue
        if is_guarded(chess, row, col + 2 * col_step, side):
            continue

        move_list.append(Move(row, king_dest))

    #Return AFTER the loop, so both sides get considered
    return move_list

def get_castling_rights(chess, side):
    if side > 0:
        if chess.white_king_castle and chess.white_queen_castle:
            return [6, 5]
        elif chess.white_queen_castle:
            return [5]
        elif chess.white_king_castle:
            return [6]
        else:
            return []
    if side < 0:
        if chess.black_king_castle and chess.black_queen_castle:
            return [6, 5]
        elif chess.black_queen_castle:
            return [5]
        elif chess.black_king_castle:
            return [6]
        else:
            return []
    return[]

def pseudo_legal_moves(chess):
    side = chess.side_to_move
    squares = chess.white_pieces if side > 0 else chess.black_pieces
    moves = []
    for r, c in squares:
        for move in available_moves(chess, r, c):
            moves.append(((r, c), move))
    return moves

#Compares two moves
def same_move(move1, move2):
    (o1, m1), (o2, m2) = move1, move2
    return (o1 == o2 and m1.row == m2.row and m1.col == m2.col
            and m1.promotion == m2.promotion)