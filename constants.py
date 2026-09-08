#Piece Codes. Sign carries side, magnitude carries type
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

#Piece name and value correlation
PIECE_NAMES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
#Pawn Promotion options
PROMOTION_PIECES = [KNIGHT, BISHOP, ROOK, QUEEN]
#Assigns values based on piece type
PIECE_VALUES = {PAWN: 100, KNIGHT: 300, BISHOP: 300, ROOK: 500, QUEEN: 900, KING: 0}

#Phase Values (Used for determining whether we are in the opening, middlegame, or endgame)
PHASE_VALUES = {PAWN: 0, KNIGHT: 300, BISHOP: 300, ROOK: 500, QUEEN: 1000, KING: 0}
MAX_PHASE = (PHASE_VALUES[KNIGHT] * 4 + PHASE_VALUES[BISHOP] * 4 + PHASE_VALUES[ROOK] * 4
             + PHASE_VALUES[QUEEN] * 2)
#List of tuples containing all knight moves, and increments for different directions
KNIGHT_MOVES = ((-2, 1), (-2, -1), (2, 1), (2, -1), (1, -2), (-1, -2), (1, 2), (-1, 2))
KING_MOVES = ((1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1))
ORTHOGONAL = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAGONAL = ((1, 1), (-1, 1), (-1, -1), (1, -1))

#Sentinel for update_piece_position: "this piece has no square"
NO_SQUARE = (-1, -1)

#MATE value for search algs
MATE = 100000

#Because python indexing wraps around, use this function to ensure the coords are on the board
def on_board(r, c):
    return 0 <= r < 8 and 0 <= c < 8

#Determines which side the piece is on and returns 0 if there is no piece
def side_of(piece):
    if piece > 0:
        return 1
    elif piece < 0:
        return -1
    return 0

#Shows the forward direction for black and white
def forward(side):
    return -side

#Finds starting row for pawns
def starting_row(side):
    return 6 if side > 0 else 1

#Finds promotion row for pawns
def promotion_row(side):
    return 0 if side > 0 else 7

#Determines whether the piece encountered is ally or enemy (takeable)
def is_enemy(piece, side):
    return piece * side < 0

#Determines whether the piece encountered is ally or enemy (takeable)
def is_friendly(piece, side):
    return piece * side > 0