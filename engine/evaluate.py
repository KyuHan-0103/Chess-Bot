from abc import ABC, abstractmethod
import torch

from core.constants import *
from engine.psqt import *
from core.board import Chess
from network.base_network import Net, K

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
    def __init__(self, weight_path : str='best.pt'):
        self.model = Net()
        #Load weights into the model
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

    def self_evaluate(self, chess : Chess):
        features = torch.tensor([self.piece_indexer(chess)])
        with torch.no_grad():
            logit = self.model(features)
        return logit.item() * K

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
