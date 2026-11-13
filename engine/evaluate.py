from abc import ABC, abstractmethod
import torch
import numpy as np

from core.constants import *
from engine.psqt import *
from core.board import Chess
from network.base_network import K

class Evaluator(ABC):

    @abstractmethod
    def self_evaluate(self, chess : Chess):
        pass

class HandEvaluator(Evaluator):
    def __init__(self):
        return
    
    def self_evaluate(self, chess : Chess):
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

class NetworkEvaluator(Evaluator):
    def __init__(self, weight_path : str='net.npz'):
        net = np.load(weight_path)
        self.w1 = net['accumulator.weight']       # (769, 256)
        self.b1 = net['acc_bias']                 # (256,)
        self.w2 = net['l1.weight']                # (32, 256)  -- (out, in)
        self.b2 = net['l1.bias']                  # (32,)
        self.w3 = net['out.weight'].ravel()       # (1, 32) -> (32,)
        self.b3 = float(net['out.bias'][0])
        assert not self.w1[768].any(), "row 768 is not zero: wrong file or wrong padding_idx"

    def self_evaluate(self, chess : Chess):
        acc = self.w1[self.piece_indexer(chess)].sum(axis=0)
        h = np.clip(acc + self.b1, 0, 1)
        h = np.clip(self.w2 @ h + self.b2, 0, 1)
        return (self.w3 @ h + self.b3) * K

    def piece_indexer(self, chess : Chess):
        #We only want to flip if it is black's turn to move
        board, bool_flip = chess.chess_board, chess.side_to_move == -1
        indices = []
        for squares in (chess.white_pieces, chess.black_pieces):
            for r, c in squares:
                piece = board[r][c]
                piece_val = abs(piece) + 5 if (piece > 0) != bool_flip else 6 - abs(piece)
                indices.append(piece_val * 64 + (7 - r if bool_flip else r) * 8 + c)
        return indices
