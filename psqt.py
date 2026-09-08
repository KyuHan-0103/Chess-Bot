"""
ORIENTATION
    Every table is written rank 8 first, so its row index lines up with
    chess_board's row index for a WHITE piece:
 
        white piece at board[r][c]  ->  table[r][c]
        black piece at board[r][c]  ->  table[7 - r][c]
 
    That mirror is why one table per piece is enough -- there is no separate
    set for black. Only the rank flips; the files do not, because White's
    a-file is Black's a-file too.
 
    You should not normally index PSQT by hand. TABLE below has the material
    value folded in, the rank already mirrored, and Black's side already
    negated, so evaluate.py can add one lookup per piece.
"""
from constants import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

MIDGAME, ENDGAME = 0, 1


"""
===========================================================================
The tables
===========================================================================
Positional bonus only -- material lives in MATERIAL below. Units are centipawns on the pawn = 100 scale.
"""

PSQT = {
  MIDGAME: {
    PAWN: [
        [    0,     0,     0,     0,     0,     0,     0,     0],   # rank 8
        [  120,   163,    74,   116,    83,   154,    41,   -13],   # rank 7
        [   -7,     9,    32,    38,    79,    68,    30,   -24],   # rank 6
        [  -17,    16,     7,    26,    28,    15,    21,   -28],   # rank 5
        [  -33,    -2,    -6,    15,    21,     7,    12,   -30],   # rank 4
        [  -32,    -5,    -5,   -12,     4,     4,    40,   -15],   # rank 3
        [  -43,    -1,   -24,   -28,   -18,    29,    46,   -27],   # rank 2
        [    0,     0,     0,     0,     0,     0,     0,     0],   # rank 1
    ],
    KNIGHT: [
        [ -204,  -109,   -41,   -60,    74,  -118,   -18,  -130],   # rank 8
        [  -89,   -50,    88,    44,    28,    76,     9,   -21],   # rank 7
        [  -57,    73,    45,    79,   102,   157,    89,    54],   # rank 6
        [  -11,    21,    23,    65,    45,    84,    22,    27],   # rank 5
        [  -16,     5,    20,    16,    34,    23,    26,   -10],   # rank 4
        [  -28,   -11,    15,    12,    23,    21,    30,   -20],   # rank 3
        [  -35,   -65,   -15,    -4,    -1,    22,   -17,   -23],   # rank 2
        [ -128,   -26,   -71,   -40,   -21,   -34,   -23,   -28],   # rank 1
    ],
    BISHOP: [
        [  -35,     5,  -100,   -45,   -30,   -51,     9,   -10],   # rank 8
        [  -32,    20,   -22,   -16,    37,    72,    22,   -57],   # rank 7
        [  -20,    45,    52,    49,    43,    61,    45,    -2],   # rank 6
        [   -5,     6,    23,    61,    45,    45,     9,    -2],   # rank 5
        [   -7,    16,    16,    32,    41,    15,    12,     5],   # rank 4
        [    0,    18,    18,    18,    17,    33,    22,    12],   # rank 3
        [    5,    18,    20,     0,     9,    26,    40,     1],   # rank 2
        [  -40,    -4,   -17,   -26,   -16,   -15,   -48,   -26],   # rank 1
    ],
    ROOK: [
        [   39,    51,    39,    62,    77,    11,    38,    52],   # rank 8
        [   33,    39,    71,    76,    98,    82,    32,    54],   # rank 7
        [   -6,    23,    32,    44,    21,    55,    74,    20],   # rank 6
        [  -29,   -13,     9,    32,    29,    43,   -10,   -24],   # rank 5
        [  -44,   -32,   -15,    -1,    11,    -9,     7,   -28],   # rank 4
        [  -55,   -30,   -20,   -21,     4,     0,    -6,   -40],   # rank 3
        [  -54,   -20,   -24,   -11,    -1,    13,    -7,   -87],   # rank 2
        [  -23,   -16,     1,    21,    20,     9,   -45,   -32],   # rank 1
    ],
    QUEEN: [
        [  -34,     0,    35,    15,    72,    54,    52,    55],   # rank 8
        [  -29,   -48,    -6,     1,   -20,    70,    34,    66],   # rank 7
        [  -16,   -21,     9,    10,    35,    68,    57,    70],   # rank 6
        [  -33,   -33,   -20,   -20,    -1,    21,    -2,     1],   # rank 5
        [  -11,   -32,   -11,   -12,    -2,    -5,     4,    -4],   # rank 4
        [  -17,     2,   -13,    -2,    -6,     2,    17,     6],   # rank 3
        [  -43,   -10,    13,     2,    10,    18,    -4,     1],   # rank 2
        [   -1,   -22,   -11,    12,   -18,   -30,   -38,   -61],   # rank 1
    ],
    KING: [
        [  -79,    28,    20,   -18,   -68,   -41,     2,    16],   # rank 8
        [   35,    -1,   -24,    -9,   -10,    -5,   -46,   -35],   # rank 7
        [  -11,    29,     2,   -20,   -24,     7,    27,   -27],   # rank 6
        [  -21,   -24,   -15,   -33,   -37,   -30,   -17,   -44],   # rank 5
        [  -60,    -1,   -33,   -48,   -56,   -54,   -40,   -62],   # rank 4
        [  -17,   -17,   -27,   -56,   -54,   -37,   -18,   -33],   # rank 3
        [    1,     9,   -10,   -78,   -52,   -20,    11,    10],   # rank 2
        [  -18,    44,    15,   -66,    10,   -34,    29,    17],   # rank 1
    ],
  },
  ENDGAME: {
    PAWN: [
        [    0,     0,     0,     0,     0,     0,     0,     0],   # rank 8
        [  217,   211,   193,   163,   179,   161,   201,   228],   # rank 7
        [  115,   122,   104,    82,    68,    65,   100,   102],   # rank 6
        [   39,    29,    16,     6,    -2,     5,    21,    21],   # rank 5
        [   16,    11,    -4,    -9,    -9,   -10,     4,    -1],   # rank 4
        [    5,     9,    -7,     1,     0,    -6,    -1,   -10],   # rank 3
        [   16,    10,    10,    12,    16,     0,     2,    -9],   # rank 2
        [    0,     0,     0,     0,     0,     0,     0,     0],   # rank 1
    ],
    KNIGHT: [
        [  -71,   -46,   -16,   -34,   -38,   -33,   -77,  -121],   # rank 8
        [  -30,   -10,   -30,    -2,   -11,   -30,   -29,   -63],   # rank 7
        [  -29,   -24,    12,    11,    -1,   -11,   -23,   -50],   # rank 6
        [  -21,     4,    27,    27,    27,    13,    10,   -22],   # rank 5
        [  -22,    -7,    20,    30,    20,    21,     5,   -22],   # rank 4
        [  -28,    -4,    -1,    18,    12,    -4,   -24,   -27],   # rank 3
        [  -51,   -24,   -12,    -6,    -2,   -24,   -28,   -54],   # rank 2
        [  -35,   -62,   -28,   -18,   -27,   -22,   -61,   -78],   # rank 1
    ],
    BISHOP: [
        [  -17,   -26,   -13,   -10,    -9,   -11,   -21,   -29],   # rank 8
        [  -10,    -5,     9,   -15,    -4,   -16,    -5,   -17],   # rank 7
        [    2,   -10,     0,    -1,    -2,     7,     0,     5],   # rank 6
        [   -4,    11,    15,    11,    17,    12,     4,     2],   # rank 5
        [   -7,     4,    16,    23,     9,    12,    -4,   -11],   # rank 4
        [  -15,    -4,    10,    12,    16,     4,    -9,   -18],   # rank 3
        [  -17,   -22,    -9,    -1,     5,   -11,   -18,   -33],   # rank 2
        [  -28,   -11,   -28,    -6,   -11,   -20,    -6,   -21],   # rank 1
    ],
    ROOK: [
        [   16,    12,    22,    18,    15,    15,    10,     6],   # rank 8
        [   13,    16,    16,    13,    -4,     4,    10,     4],   # rank 7
        [    9,     9,     9,     6,     5,    -4,    -6,    -4],   # rank 6
        [    5,     4,    16,     1,     2,     1,    -1,     2],   # rank 5
        [    4,     6,    10,     5,    -6,    -7,   -10,   -13],   # rank 4
        [   -5,     0,    -6,    -1,    -9,   -15,   -10,   -20],   # rank 3
        [   -7,    -7,     0,     2,   -11,   -11,   -13,    -4],   # rank 2
        [  -11,     2,     4,    -1,    -6,   -16,     5,   -24],   # rank 1
    ],
    QUEEN: [
        [  -11,    27,    27,    33,    33,    23,    12,    24],   # rank 8
        [  -21,    24,    39,    50,    71,    30,    37,     0],   # rank 7
        [  -24,     7,    11,    60,    57,    43,    23,    11],   # rank 6
        [    4,    27,    29,    55,    70,    49,    70,    44],   # rank 5
        [  -22,    34,    23,    57,    38,    41,    48,    28],   # rank 4
        [  -20,   -33,    18,     7,    11,    21,    12,     6],   # rank 3
        [  -27,   -28,   -37,   -20,   -20,   -28,   -44,   -39],   # rank 2
        [  -40,   -34,   -27,   -52,    -6,   -39,   -24,   -50],   # rank 1
    ],
    KING: [
        [  -90,   -43,   -22,   -22,   -13,    18,     5,   -21],   # rank 8
        [  -15,    21,    17,    21,    21,    46,    28,    13],   # rank 7
        [   12,    21,    28,    18,    24,    55,    54,    16],   # rank 6
        [  -10,    27,    29,    33,    32,    40,    32,     4],   # rank 5
        [  -22,    -5,    26,    29,    33,    28,    11,   -13],   # rank 4
        [  -23,    -4,    13,    26,    28,    20,     9,   -11],   # rank 3
        [  -33,   -13,     5,    16,    17,     5,    -6,   -21],   # rank 2
        [  -65,   -41,   -26,   -13,   -34,   -17,   -29,   -52],   # rank 1
    ],
  }
}

"""
===========================================================================
Material
===========================================================================
Phase dependent on purpose: a pawn grows as the board empties, a knight
shrinks, a rook grows.

These are deliberately NOT constants.PIECE_VALUES. Those stay as they are for
MVV-LVA in ordering.py, where only the relative order matters. These are the
values the tables above were tuned against, so using them here keeps the two
halves of the evaluation consistent. Two value sets for two different jobs.
"""

MATERIAL = {
    MIDGAME: {PAWN: 100, KNIGHT: 411, BISHOP: 445, ROOK: 582, QUEEN: 1250, KING: 0},
    ENDGAME: {PAWN: 115, KNIGHT: 343, BISHOP: 362, ROOK: 624, QUEEN: 1141, KING: 0},
}

"""
===========================================================================
The combined lookup
===========================================================================
TABLE[phase][piece][row][col] -> material + placement, ready to add.

The key is the SIGNED board value, the thing chess_board already holds, not
the piece type. -2 and 2 are separate entries. That is what lets the mirror
live inside the table: a black piece standing on row r gets its entry built
from row 7 - r, so the caller never flips anything and never calls abs().

Values stay positive for both colours. An entry is what the piece is worth
on that square, not its contribution to the score, so the evaluation still
adds for white and subtracts for black.
"""

def _build_table():
    table = {}
    for phase in (MIDGAME, ENDGAME):
        table[phase] = {}
        for piece_type in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):
            material = MATERIAL[phase][piece_type]
            placement = PSQT[phase][piece_type]
            for side in (1, -1):
                rows = []
                for row in range(8):
                    #row is where the piece stands, look is the row it reads
                    look = row if side > 0 else 7 - row
                    rows.append([material + value for value in placement[look]])
                table[phase][piece_type * side] = rows
    return table


TABLE = _build_table()