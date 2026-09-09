from core.constants import *
from core.movegen import *
from core.attacks import *
from core.zobrist import *

"""
===========================================================================
Making and unmaking moves
===========================================================================
"""
def make_move(chess, origin, target):
    """
    Apply a move to the board. Returns true if it is legal false if not

    An illegal move (one that leaves the mover's own king attacked) is undone
    before returning False, so the caller can treat make_move as a filter:
        if not make_move(chess, origin, move):
            continue
    """
    if chess.game is not None:
        return False
    
    board = chess.chess_board
    from_row, from_col = origin
    to_row, to_col, promotion = target
    moving_piece = board[from_row][from_col]
    captured_piece = board[to_row][to_col]
    side = side_of(moving_piece)

    #Guards against making a move with a piece at 0 and calling is guarded with side = 0
    if moving_piece == 0:
        raise ValueError(f"no piece at {origin}")

    """-----------------------------Work out what kind of move this is-----------------------------"""
    #A pawn moving diagonally onto an EMPTY square can only be en passant.
    #The captured pawn is beside the destination, not on it.
    ep_capture = None
    if abs(moving_piece) == PAWN and to_col != from_col and captured_piece == 0:
        ep_capture = (from_row, to_col)

    #A king moving two files can only be castling
    castle_rook = None
    if abs(moving_piece) == KING and abs(to_col - from_col) == 2:
        castle_rook = (from_row, 7, from_row, 5) if to_col > from_col else (from_row, 0, from_row, 3)

    is_promotion = abs(moving_piece) == PAWN and to_row == promotion_row(side)
    if is_promotion and promotion == 0:
        raise ValueError(f"promotion move to {(to_row, to_col)} carries no piece type")
    placed_piece = promotion * side if promotion else moving_piece
    """------------------Save everything the move destroys, before anything changes------------------"""

    chess.move_order.append((
        moving_piece, from_row, from_col,
        captured_piece, to_row, to_col,
        ep_capture, castle_rook,
        dict(chess.en_passant),
        (chess.white_king_castle, chess.white_queen_castle,
         chess.black_king_castle, chess.black_queen_castle),
         chess.half_move_clock, chess.game, chess.zobrist, chess.phase))

    """------------------Move the pieces, on the board and in the piece sets together------------------"""
    relocate(chess, placed_piece, captured_piece, from_row, from_col,
             to_row, to_col, ep_capture, castle_rook, side)

    """---------------------Update the state derived from where the pieces now are---------------------"""
    #Second call catches a rook captured on its own home square
    #Zobrist hash is updated in these functions
    update_castling_rights(chess, moving_piece, from_row, from_col)
    update_castling_rights(chess, captured_piece, to_row, to_col)

    if abs(moving_piece) == KING:
        update_king_pos(chess, to_row, to_col, side)

    refresh_en_passant(chess, moving_piece, from_row, to_row, to_col, side)

    """----------------------Bring the hash up to date for the pieces that moved----------------------"""
    rehash_pieces(chess, moving_piece, placed_piece, captured_piece, from_row, from_col,
                  to_row, to_col, ep_capture, castle_rook, side)
    #Updates phase score
    #Doesn't have to consider ep because pawns are valued at 0
    if captured_piece:
        chess.phase -= PHASE_VALUES[abs(captured_piece)]
    if placed_piece != moving_piece:
        chess.phase += PHASE_VALUES[abs(placed_piece)] - PHASE_VALUES[abs(moving_piece)]
    """---------------------------------------Advance the clocks---------------------------------------"""
    chess.play += 1
    chess.full_move_count = 1 + chess.play // 2
    chess.side_to_move *= -1
    toggle_side(chess)
    chess.half_move_clock += 1

    #Counted BEFORE the legality check, so undo_move can always decrement
    #symmetrically without needing to know whether the move was legal
    key = chess.zobrist
    chess.position_counts[chess.zobrist] = chess.position_counts.get(key, 0) + 1

    """---------------------Legality: a move that exposes your king never happened---------------------"""   
    if in_check(chess, side):
        undo_move(chess)
        return False
    
    """-----------------------Draw detection, only for moves that actually saved-----------------------"""   
    if chess.position_counts[key] >= 3:
        chess.game = "Draw by repetition"
    update_fifty_move_rule(chess, moving_piece, captured_piece)
    return True

#Moves pieces on the board and in the piece sets. These two representations
#have to change together or pseudo_legal_moves goes stale.
def relocate(chess, placed_piece, captured_piece, from_row, from_col,
             to_row, to_col, ep_capture, castle_rook, side):
    
    board = chess.chess_board

    #The mover. placed_piece differs from moving_piece only on a promotion.
    board[to_row][to_col] = placed_piece
    board[from_row][from_col] = 0
    update_piece_position(chess, (from_row, from_col), (to_row, to_col), side)

    #Anything standing on the destination is gone
    if captured_piece != 0:
        update_piece_position(chess, (to_row, to_col), NO_SQUARE, -side)

    #An en passant victim sits beside the destination, not on it
    if ep_capture is not None:
        board[ep_capture[0]][ep_capture[1]] = 0
        update_piece_position(chess, (ep_capture[0], ep_capture[1]), NO_SQUARE, -side)

    #Castling moves a second piece of the same side
    if castle_rook is not None:
        r0, c0, r1, c1 = castle_rook
        board[r1][c1] = board[r0][c0]
        board[r0][c0] = 0
        update_piece_position(chess, (r0, c0), (r1, c1), side)

#Rebuilds chess.en_passant for the position the mover just created, and keeps
#the hash in step. Only a pawn double step can create a capture opportunity.
def refresh_en_passant(chess, moving_piece, from_row, to_row, to_col, side):
    board = chess.chess_board
 
    #XOR the OLD file out before discarding it
    if chess.en_passant:
        toggle_en_passant(chess, next(iter(chess.en_passant.values())).col)
    chess.en_passant.clear()
 
    if abs(moving_piece) != PAWN or abs(from_row - to_row) != 2:
        return
 
    for col_add in (1, -1):
        new_col = to_col + col_add
        if on_board(to_row, new_col) and board[to_row][new_col] == PAWN * -side:
            chess.en_passant[(to_row, new_col)] = Move(
                to_row + forward(-side), to_col)
 
    #Only hash it when a capture is actually available, matching full_hash
    if chess.en_passant:
        toggle_en_passant(chess, to_col)

#Every piece-square change the move made, XORed into the hash.
def rehash_pieces(chess, moving_piece, placed_piece, captured_piece,
                  from_row, from_col, to_row, to_col, ep_capture,
                  castle_rook, side):
    #Lift the mover off its origin
    toggle_piece(chess, moving_piece, from_row, from_col)
    #Remove whatever stood on the destination
    if captured_piece:
        toggle_piece(chess, captured_piece, to_row, to_col)
    #Set the mover (or its promoted form) down
    toggle_piece(chess, placed_piece, to_row, to_col)

    #If en_passant remove the captured pawn
    if ep_capture is not None:
        toggle_piece(chess, PAWN * -side, ep_capture[0], ep_capture[1])

    #If castles move the rook
    if castle_rook is not None:
        r0, c0, r1, c1 = castle_rook
        toggle_piece(chess, ROOK * side, r0, c0)
        toggle_piece(chess, ROOK * side, r1, c1)
 
def undo_move(chess):
    """Reverses make_move exactly, including the piece sets and the hash."""
    (moving_piece, from_row, from_col,
     captured_piece, to_row, to_col,
     ep_capture, castle_rook, prev_en_passant, prev_rights,
     prev_half_move_clock, prev_game, prev_zobrist, prev_phase) = chess.move_order.pop()
    board = chess.chess_board
    side = side_of(moving_piece)
 
    #FIRST, before the board changes: the key describes the position we are
    #leaving, so it cannot be computed after the pieces move back
    key = chess.zobrist
    remaining = chess.position_counts.get(key, 0) - 1
    if remaining <= 0:
        chess.position_counts.pop(key, None)
    else:
        chess.position_counts[key] = remaining
 
    #RESTORED, not recomputed
    chess.zobrist = prev_zobrist
    chess.phase = prev_phase

    #-- Board and piece sets, mirroring relocate() in reverse -------------
    if castle_rook is not None:
        r0, c0, r1, c1 = castle_rook
        board[r0][c0] = board[r1][c1]
        board[r1][c1] = 0
        update_piece_position(chess, (r1, c1), (r0, c0), side)
 
    board[from_row][from_col] = moving_piece
    board[to_row][to_col] = captured_piece
    update_piece_position(chess, (to_row, to_col), (from_row, from_col), side)
 
    if captured_piece != 0:
        update_piece_position(chess, NO_SQUARE, (to_row, to_col), -side)
 
    if ep_capture is not None:
        board[ep_capture[0]][ep_capture[1]] = PAWN * -side
        update_piece_position(chess, NO_SQUARE, ep_capture, -side)
 
    #-- Derived state ----------------------------------------------------
    if abs(moving_piece) == KING:
        update_king_pos(chess, from_row, from_col, side)
 
    chess.en_passant.clear()
    chess.en_passant.update(prev_en_passant)
    (chess.white_king_castle, chess.white_queen_castle,
     chess.black_king_castle, chess.black_queen_castle) = prev_rights
 
    chess.half_move_clock = prev_half_move_clock
    chess.game = prev_game
    chess.play -= 1
    chess.full_move_count = 1 + chess.play // 2
    chess.side_to_move *= -1

"""Updating castling and piece positions"""
#Updates castling rights if the kning moves or if the rook moves
#Updates zobrist hash as well
def update_castling_rights(chess, piece, row, col):
    #If king moves, both sides (king and queen sides) lose their rights
    if abs(piece) == KING:
        if piece > 0:
            #Only update if field and zobrist hash if there is a change
            if chess.white_king_castle:
                toggle_castle_rights(chess, 0)
                chess.white_king_castle = False
            if chess.white_queen_castle:
                toggle_castle_rights(chess, 1)
                chess.white_queen_castle = False
        else:
            if chess.black_king_castle:
                toggle_castle_rights(chess, 2)
                chess.black_king_castle = False
            if chess.black_queen_castle:
                toggle_castle_rights(chess, 3)
                chess.black_queen_castle = False
    #If a rook move, only the king or queen side loses its right
    elif abs(piece) == ROOK:
        if piece > 0 and row == 7:
            if col == 7 and chess.white_king_castle:
                toggle_castle_rights(chess, 0)
                chess.white_king_castle = False
            elif col == 0 and chess.white_queen_castle:
                toggle_castle_rights(chess, 1)
                chess.white_queen_castle = False
        elif piece < 0 and row == 0:
            if col == 7 and chess.black_king_castle:
                toggle_castle_rights(chess, 2)
                chess.black_king_castle = False
            elif col == 0 and chess.black_queen_castle:
                toggle_castle_rights(chess, 3)
                chess.black_queen_castle = False      

#Updates king position when a king moves
def update_king_pos(chess, row, col, side):
    if side > 0:
        chess.white_king_pos = (row, col)
    else:
        chess.black_king_pos = (row, col)

#Promotes pawn to inputted piece type (defaults to QUEEN)
def promote(chess, row, col, side, piece_type=QUEEN):
    chess.chess_board[row][col] = piece_type * side

#NO_SQUARE as the target removes a piece; NO_SQUARE as the origin adds one.
#remove() rather than discard() is deliberate: a KeyError here means the sets
#have drifted out of step with the board, and you want to hear about it now.
def update_piece_position(chess, origin, target, side):
    if side > 0:
        squares = chess.white_pieces
    elif side < 0:
        squares = chess.black_pieces
    else:
        return

    if origin[0] >= 0:
        squares.remove(origin)
    if target[0] >= 0:
        squares.add(target)

"""Game state moves: Checks for for game results and draws"""

#Checks for fifty move rule
def update_fifty_move_rule(chess, og_piece, tar_piece):
    if abs(og_piece) == PAWN or tar_piece != 0:
        chess.half_move_clock = 0
    elif chess.half_move_clock >= 100:
        chess.game = "Draw by fifty move rule"

#Does this side have any move that does not leave its own king in check?
#Stops at the first one, so it is far cheaper than generating them all.
def has_legal_move(chess):
    for origin, target in legal_move_list(chess):
        if make_move(chess, origin, target):
            undo_move(chess)
            return True
    return False

#Neither side has enough material left to force a mate, so the game is drawn
#no matter how long it goes on.
#
#The piece count is the gate, and it is O(1) on sets we already maintain
#incrementally, so this costs one addition and one comparison in the
#overwhelming majority of positions. chess.phase looks like a cheaper gate but
#is the wrong one: PHASE_VALUES[PAWN] is 0, so phase == 0 is true of any
#pawns-and-kings position, which is emphatically not a draw.
#
#The list is deliberately conservative. KBN vs K is a forced win and KNN vs K
#is only a draw with best play, so neither belongs here: claiming a draw that
#is not one corrupts a result, while missing one merely costs a little time.
def insufficient_material(chess):
    if len(chess.white_pieces) + len(chess.black_pieces) > 4:
        return False

    board = chess.chess_board
    #Bishops are kept as square colours, because two bishops drawn only when
    #they are on the same colour
    minors = {1: [], -1: []}
    for row, col in chess.white_pieces | chess.black_pieces:
        piece = board[row][col]
        kind = abs(piece)
        if kind == KING:
            continue
        #A single pawn, rook or queen is already enough to mate with
        if kind in (PAWN, ROOK, QUEEN):
            return False
        minors[side_of(piece)].append((row + col) % 2 if kind == BISHOP else None)
 
    white, black = minors[1], minors[-1]
    #K v K, and K + one minor v K either way
    if len(white) + len(black) <= 1:
        return True
    #K + B v K + B, drawn only with both bishops on one colour
    if len(white) == 1 and len(black) == 1:
        return white[0] is not None and white[0] == black[0]
    return False
#None while the game is live, otherwise the reason it ended.
#Call this AFTER a move has been made, to judge the position the mover left.
def game_result(chess):
    if chess.game is not None:
        return chess.game                      # repetition or fifty move, already set
    if insufficient_material(chess):
        return "Draw by insufficient material"
    if has_legal_move(chess):
        return None
    if in_check(chess, chess.side_to_move):
        return "Checkmate"
    return "Draw by stalemate"
