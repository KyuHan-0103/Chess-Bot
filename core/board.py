from collections import deque
from core.zobrist import *
from core.constants import *

class Chess:
    def __init__(self):
        self.chess_board = [
            [-4, -2, -3, -5, -6, -3, -2, -4],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [4, 2, 3, 5, 6, 3, 2, 4]
            ]
        #Move_Order contains a tuple of 6 ints per element, a dict and a tuple
        #(piece_type at origin, origin_row, origin_column, piece_type at target, target_row, target_column)
        #Dict full of potential en passant moves
        #Tuple holding booleans of castling rights
        self.move_order = deque()

        #Dictionary holding a key (full board position) and value (how many repetitions)
        self.position_counts = {}

        #Full move count tracks the total number of turns (both players have made)
        #One full turn is both players moving once
        self.full_move_count = 1
        self.play = 0
        #Positive means that it is white's turn negative means black's turn
        self.side_to_move = 1

        #Half moves (just one person's turn) is tracked for 
        self.half_move_clock = 0


        #Castling rights for black and white (for both queen and king side castling)
        #These fields become false if:
        # 1. The king has moved
        # 2. The rook has moved (king side would become false if kingside rook moved, but queenside would remain true)

        self.white_king_castle = True
        self.white_queen_castle = True

        self.black_king_castle = True
        self.black_queen_castle = True

        #Contains sqaure coordinates containing squares which if they contain a pawn can en passant (as keys)
        #Values are the place they can en passant to
        self.en_passant = {}

        self.black_king_pos = (0, 4)
        self.white_king_pos = (7, 4)

        #None while the game is live, otherwise the reason it ended
        self.game = None

        """Review zobrist implementation"""
        self.zobrist = full_hash(self)

        self.white_pieces = {(r, c) for r in range(8) for c in range(8)
                             if self.chess_board[r][c] > 0}
        self.black_pieces = {(r, c) for r in range(8) for c in range(8)
                             if self.chess_board[r][c] < 0}

        self.phase = self.count_phase()

    #Derived from the board for the same reason the piece sets and the zobrist
    #hash are. Hardcoding 6400 is only right for the starting array and
    #silently wrong for any position built some other way, so anything that
    #installs a new board (from_fen, a loaded game, a test fixture) has to
    #call this the same way it already rebuilds the hash.
    def count_phase(self):
        return sum(PHASE_VALUES[abs(self.chess_board[r][c])]
                   for r, c in self.white_pieces | self.black_pieces)

