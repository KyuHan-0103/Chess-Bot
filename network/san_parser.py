import re
from core.movegen import available_moves, Move
from core.rules import make_move, undo_move
from core.constants import MATE
"""SAN-CONSTANTS"""
PAWN = 1
PIECE_KEYS = {"N" : 2, "B" : 3, "R" : 4, "Q" : 5, "K" : 6}
FILE_KEYS = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7}

#if move[0] not in PIECE_KEYS
# piece_loc = PAWN

"""Prepares SAN move text to only contain moves
Should be: e4 e5 Nc3 etc...
NOTE: this text prep does not remove the x which denotes captures 
(in case we want to only look at moves with captures)
"""
_TOKEN = re.compile(r"""
      (?P<comment>\{[^}]*\})          # { ... } comment, may span lines
    | (?P<line_comment>;[^\n]*)       # ; comment to end of line
    | (?P<open>\()                    # start of a side variation
    | (?P<close>\))                   # end of a side variation
    | (?P<result>1-0|0-1|1/2-1/2|\*)  # game result
    | (?P<number>\d+\.+)              # move number: "12." or "12..."
    | (?P<nag>\$\d+|[!?]+)            # $1-style glyph, or a detached ! / ?
    | (?P<move>[^\s{}();$]+)          # anything else is a move; the board validates it
    | (?P<bad>\S)                     # unclosed {, stray }, lone $
""", re.VERBOSE)

_EVAL = re.compile(r"\[%eval\s*(#?)([+-]?\d+(?:\.\d+)?)")


def san_text_prep_with_eval(source: str) -> list[tuple[str, float | None]]:
    """Every main-line move in order, paired with the eval from the comment
    after it, or None if there's no [%eval] there.
    convertes eval to be from the mover's perspective
    '1. e4 { [%eval 0.17] } 1... e5 2. Nf3 { [%eval 0.2] }'
        -> [('e4', 17), ('e5', None), ('Nf3', 20)]
    """
    pairs: list[tuple[str, float | None]] = []
    depth = 0  # > 0 while inside ( ... ) side variations

    for tok in _TOKEN.finditer(source):
        kind, text = tok.lastgroup, tok.group()

        if kind == "open":
            depth += 1
        elif kind == "close":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unmatched ')' at offset {tok.start()}")
        elif kind == "bad":
            raise ValueError(f"malformed movetext: {text!r} at offset {tok.start()}")
        elif depth:
            continue  # side-line moves and comments must never reach the main line
        elif kind == "move":
            # Every move gets a slot, eval or not. Its eval, if any, arrives in the next comment.
            pairs.append((text.rstrip("+#?!"), None))
        elif kind == "comment" and pairs:
            found = _EVAL.search(text)
            if found:
                #pairs already holds the move this comment follows. White plays
                #moves 1, 3, 5, ... so an odd count means White just moved
                #(assumes the normal starting position, White first)
                mover = 1 if len(pairs) % 2 else -1
                pairs[-1] = (pairs[-1][0], eval_to_number(*found.groups(), mover=mover))
        # number / result / nag / line_comment: nothing to record

    if depth:
        raise ValueError("unclosed '(' in movetext")
    return pairs

def eval_to_number(mate_flag : str, number : str, mover : int | None = None) -> float:
    """Centipawns from White's side: '-0.27' -> -27, '#3' -> MATE - 3, '#-3' -> -(MATE - 3)

    mover is +1 if White made the move this eval follows, -1 if Black. Only '#0'
    needs it: that's a mate already on the board, delivered by whoever just
    moved, and the digits carry no sign to say who.
    """
    if not mate_flag:
        #round() keeps centipawns exact: float('0.57') * 100 is 56.99999999999999
        return round(float(number) * 100) * mover
    mate_in = int(number)
    if mate_in == 0:
        sign = mover
    else:
        sign = 1 if mate_in > 0 else -1
    return sign * (MATE - abs(mate_in))

"""NOTE: This function takes ONE move at a time with no extra spaces."""
def san_make_move(chess, move_text : str):
    side_to_move = chess.side_to_move
    #Checks for castling and makes the castles move if so
    if move_text[0] == 'O':
        san_castles(chess, move_text, side_to_move)
        return

    #Removes x
    move_text = move_text.replace('x', '')

    #Checks for promotions
    promotion = 0
    if '=' in move_text:
        #Grabs collection promotion piece if promotion exists
        promotion : int = PIECE_KEYS[move_text[len(move_text) - 1]]
        #Slice off the promotion portion of the move
        move_text = move_text[:len(move_text) - 2]
    
    #Grabs last char which denotes row and second to last which denotes column
    target = (8 - int(move_text[-1]), FILE_KEYS[move_text[-2]])

    """Locating which piece is moving and from where"""
    from_text = move_text[:len(move_text) - 2]
    from_row = None
    from_col = None
    #If no indicator we default to a pawn
    piece_type = 1
    if len(from_text) != 0:
        #Get corresponding piece value defaulting to a pawn
        piece_type = PIECE_KEYS.get(from_text[0], 1)

        #Remove piece indicator
        if piece_type != 1:
            from_text = from_text[1:]
        
        #If we are given TWO extra coordinates, we know the first represents row, second represents column
        if len(from_text) > 1:
            from_col = FILE_KEYS[from_text[0]]
            from_row = 8 - int(from_text[1])
        elif len(from_text) > 0:
            if from_text[0] in FILE_KEYS:
                from_col = FILE_KEYS[from_text[0]]
            else:
                from_row = 8 - int(from_text[0])

    piece_location = search_for_piece(chess, piece_type, target, side_to_move, from_col, from_row)
    if not make_move(chess, piece_location, Move(target[0], target[1], promotion)):
        raise ValueError('Move failed to be played')
    return (piece_location, Move(target[0], target[1]))

"""
Searches for the correct piece to make a capture
Returns the square where the correct piece is located in
"""
def search_for_piece(chess, piece_type : int, square : tuple[int, int], side : int, column=None, row=None):
    board = chess.chess_board
    piece_locations = chess.white_pieces if side > 0 else chess.black_pieces
    column_filter = [column] if column is not None else [0, 1, 2, 3, 4, 5, 6, 7]
    row_filter = [row] if row is not None else [0, 1, 2, 3, 4, 5, 6, 7]

    res : list[tuple] = []
    #Update piece_type to reflect correct side
    piece_type *= side
    for piece_loc in list(piece_locations):
        if piece_loc[0] not in row_filter or piece_loc[1] not in column_filter:
            continue
        if board[piece_loc[0]][piece_loc[1]] == piece_type:
            #If this piece location holds the correct kind of piece, search all its moves
            #and see if it can make the correct capture
            for move in available_moves(chess, piece_loc[0], piece_loc[1]):
                if move.row == square[0] and move.col == square[1]:
                    if make_move(chess, piece_loc, move):
                        undo_move(chess)
                        #Use this check for pawn promotions (a pawn promotion will produce 4 identical moves)
                        if piece_loc not in res:
                            res.append(piece_loc)
    if len(res) == 0:
        raise ValueError('No move found')
    if len(res) > 1:
        raise ValueError('Ambiguous Position')
    return res[0]


def san_castles(chess, move_text : str, side : int):
    board = chess.chess_board
    #Checks if queenside or kingside
    if len(move_text) < 5:
        #Kingside: O-O
        if side > 0:
            if not make_move(chess, (7, 4), Move(7, 6)):
                raise ValueError('Illegal move')
        else:
            if not make_move(chess, (0, 4), Move(0, 6)):
                raise ValueError('Illegal move')
    else:
        ##Queenside: O-O-O
        if side > 0:
            if not make_move(chess, (7, 4), Move(7, 2)):
                raise ValueError('Illegal move')
        else:
            if not make_move(chess, (0, 4), Move(0, 2)):
                raise ValueError('Illegal move')