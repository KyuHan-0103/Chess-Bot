from core.board import Chess
from ui.gui import play_game        

def main() -> None:
    chess_game = Chess()
    play_game(chess_game)

if __name__ == "__main__":
    main()