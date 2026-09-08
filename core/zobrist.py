from core.constants import *
import random

random.seed(0xC0FFEE) #Fixed seed so games are reproducible

PIECE_KEYS = [[[random.getrandbits(64) for _ in range(8)]
               for _ in range(8)] for _ in range(12)]

SIDE_KEY = random.getrandbits(64)
"""
Castling Keys are in the following order: 
white kingside, white queenside, black kingside, black queenside
"""
CASTLING_KEYS = [random.getrandbits(64) for _ in range(4)]
EN_PASSANT_FILE_KEY = [random.getrandbits(64) for _ in range(8)]

def piece_index(piece):
    """converts pieces on a -6 to 6+ scale to 0 ... 11 scale"""
    return (abs(piece) - 1) + (0 if piece > 0 else 6)

def full_hash(chess):
    """Compute from scratch. Once at startup then update by move"""
    h = 0
    for r, row in enumerate(chess.chess_board):
        for c, piece in enumerate(row):
            if piece:
                h ^= PIECE_KEYS[piece_index(piece)][r][c]

    if chess.side_to_move < 0:
        h ^= SIDE_KEY

    for i, can_castle in enumerate((chess.white_king_castle, chess.white_queen_castle,
                                    chess.black_king_castle, chess.black_queen_castle)):
        if can_castle:
            h ^= CASTLING_KEYS[i]
    if chess.en_passant:
        destination = next(iter(chess.en_passant.values()))
        h ^= EN_PASSANT_FILE_KEY[destination.col]
    
    return h

"""
Review zobrist function: directly updates chess.zobrist field
"""
def toggle_piece(chess, piece, row, col):
    chess.zobrist ^= PIECE_KEYS[piece_index(piece)][row][col]

def toggle_en_passant(chess, col):
    chess.zobrist ^= EN_PASSANT_FILE_KEY[col]

def toggle_castle_rights(chess, key):
    chess.zobrist ^= CASTLING_KEYS[key]

def toggle_side(chess):
    chess.zobrist ^= SIDE_KEY