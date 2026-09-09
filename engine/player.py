from abc import ABC, abstractmethod
import random
 
from core.movegen import pseudo_legal_moves
from core.rules import make_move, undo_move
from engine.search import alpha_beta_engine
 
class Player(ABC):
    """
    Anything that can choose a move.
 
    The contract is deliberately narrow, so that a search, a random mover and
    an eventual neural net are interchangeable in the arena:
 
        select_move(chess) -> (origin, Move), or None if there is no legal move
 
    Two rules every implementation has to keep:
 
        1. It is handed the LIVE board and must leave it exactly as it found
           it. Anything that makes a move in order to look at it undoes it.
        2. None means "this position has no legal move", i.e. the game is
           over. It never means "I ran out of time". A player that cannot
           decide still has to return something legal.
    """
    @abstractmethod
    def select_move(self, board):
        """Choose a move for the side to move, without changing the board."""

    def new_game(self):
        """Forget anything learned from previous game"""
        pass

    #What this player thought of the position it last moved in, in centipawns
    #from ITS OWN point of view. The arena uses it for score based
    #adjudication; a player with no opinion returns None and is simply not
    #consulted.
    def last_score(self):
        return None
 
    #Nodes spent on the last move, for the arena's per-game totals
    def last_nodes(self):
        return 0
 

class RandomPlayer(Player):

    def __init__(self, seed=None, name="random"):
        #Its own Random instance, so a match is reproducible without
        #disturbing the module level random that anything else may be using
        self.rng = random.Random(seed)
        self.name = name

    def select_move(self, chess):
        #pseudo_legal_moves does not filter pins, so a move is only known to
        #be legal once make_move has accepted it. Terminates when there are no
        #legal moves
        moves = pseudo_legal_moves(chess)
        self.rng.shuffle(moves)
        for origin, move in moves:
            if make_move(chess, origin, move):
                undo_move(chess)
                return (origin, move)
        return None

class SearchPlayer(Player):
    """Alpha-beta search under a depth, node and/or time budget."""
    def __init__(self, depth, nodes=None, time=None, engine=None, name=None):
        self.depth = depth
        self.nodes = nodes
        self.time = time
        #Owning an engine by default is what keeps two SearchPlayers from
        #sharing a transposition table. Pass one in only to share deliberately.
        self.engine = engine if engine is not None else alpha_beta_engine()
        self.name = name or f"search(d{depth}, n={nodes}, t={time})"
        self.score = None
        self.node_count = 0

    def new_game(self):
        self.engine.new_game()
        self.score = None
        self.node_count = 0
    
    def last_score(self):
        return self.score
 
    def last_nodes(self):
        return self.node_count
 
    def select_move(self, chess):
        move, score = self.engine.search(chess, self.depth, self.nodes, self.time)
        self.node_count = self.engine.node_count()
        #A score is only meaningful alongside a move. With no legal move the
        #search returns -inf, which would poison score based adjudication.
        self.score = score if move is not None else None
        return move