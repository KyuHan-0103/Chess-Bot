from abc import ABC, abstractmethod
import random
from engine.search import search
class Player(ABC):
    @abstractmethod
    def select_move(self, board):
        """Every player must have a way to choose a move"""
        pass

class RandomPlayer(Player):
    def select_move(self, move_list):
        #Randomly selects a move
        return random.choice(move_list)

class SearchPlayer(Player):
    def __init__(self, depth, nodes, time):
        self.depth = depth
        self.nodes = nodes
        self.time = time

    def select_move(self, chess):
        #Runs neg-max /alpha-beta search
        return search(chess, self.depth, self.nodes, self.time)