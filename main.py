from board import Chess
from gui import play_game        

def main() -> None:
    chess_game = Chess()
    play_game(chess_game)

if __name__ == "__main__":
    main()