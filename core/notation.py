from core.constants import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, UCI_PROMOTION, COLUMN_CODE
from core.zobrist import full_hash
from core.board import Chess
from core.movegen import Move

# ---------------------------------------------------------------------------
# FEN parsing
# ---------------------------------------------------------------------------

FEN_PIECES = {"p": PAWN, "n": KNIGHT, "b": BISHOP,
              "r": ROOK, "q": QUEEN, "k": KING}

def from_fen(fen):
    """Build a Chess object from a FEN string."""
    parts = fen.split()
    if len(parts) < 3:
        raise ValueError(f"FEN needs at least 3 fields, got {len(parts)}: {fen!r}")

    placement, side_field, castle_field = parts[0], parts[1], parts[2]
    ep_field = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    #Clamped because a 0 (or missing) full move number would put play
    #below the start of the game
    fullmove = max(int(parts[5]), 1) if len(parts) > 5 else 1

    board = [[0] * 8 for _ in range(8)]
    row = 0
    col = 0
    for ch in placement:
        if ch == "/":
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            piece = FEN_PIECES[ch.lower()]
            board[row][col] = piece if ch.isupper() else -piece
            col += 1

    chess = Chess()
    chess.chess_board = board
    chess.side_to_move = 1 if side_field == "w" else -1

    chess.white_king_castle = "K" in castle_field
    chess.white_queen_castle = "Q" in castle_field
    chess.black_king_castle = "k" in castle_field
    chess.black_queen_castle = "q" in castle_field

    # Locate the kings rather than trusting a hardcoded square
    chess.white_king_pos = None
    chess.black_king_pos = None
    for r in range(8):
        for c in range(8):
            if board[r][c] == KING:
                chess.white_king_pos = (r, c)
            elif board[r][c] == -KING:
                chess.black_king_pos = (r, c)
    if chess.white_king_pos is None or chess.black_king_pos is None:
        raise ValueError("FEN is missing a king")

    # This engine stores en passant as {capturing_pawn_square: destination Move}
    chess.en_passant = {}
    if ep_field != "-":
        ep_col = ord(ep_field[0]) - ord("a")
        ep_row = 8 - int(ep_field[1])
        side = chess.side_to_move
        pawn_row = ep_row + side          # square the capturing pawn sits on
        if 0 <= pawn_row < 8:
            for delta in (-1, 1):
                c = ep_col + delta
                if 0 <= c < 8 and board[pawn_row][c] == PAWN * side:
                    chess.en_passant[(pawn_row, c)] = Move(ep_row, ep_col)

    chess.half_move_clock = halfmove
    chess.position_counts = {}
    chess.game = None
    #play is the ply count since the start of the game. full_move_count is
    #derived from it here exactly the way make_move and undo_move derive it,
    #so a position built from a FEN advances its clocks like a played one
    chess.play = 2 * (fullmove - 1) + (0 if chess.side_to_move > 0 else 1)
    chess.full_move_count = 1 + chess.play // 2
    chess.move_order.clear()
    chess.zobrist = full_hash(chess)
    #The piece sets were built from the start position in __init__
    chess.white_pieces = {(r, c) for r in range(8) for c in range(8)
                          if board[r][c] > 0}
    chess.black_pieces = {(r, c) for r in range(8) for c in range(8)
                          if board[r][c] < 0}
    #After the piece sets, because it counts over them
    chess.phase = chess.count_phase()
    return chess


def square_name(row, col):
    return "abcdefgh"[col] + str(8 - row)


def move_name(origin, move):
    name = square_name(*origin) + square_name(move.row, move.col)
    if move.promotion:
        name += "nbrq"[[KNIGHT, BISHOP, ROOK, QUEEN].index(move.promotion)]
    return name


# ---------------------------------------------------------------------------
# FEN writing
# ---------------------------------------------------------------------------

#The other half of FEN_PIECES: piece type -> the letter FEN uses for it.
#Case carries the side, the same way the sign of a piece code does.
FEN_LETTERS = {PAWN: "P", KNIGHT: "N", BISHOP: "B",
               ROOK: "R", QUEEN: "Q", KING: "K"}


def to_fen(chess):
    """
    Write a Chess object out as a FEN string. The inverse of from_fen.

    One asymmetry is worth knowing about: this engine only stores an en
    passant square when a capture is actually available (refresh_en_passant
    and full_hash both take that view, so two positions that differ only by
    an unusable en passant square are the same position here). So a FEN
    carrying an en passant square no pawn can use comes back out as "-".
    from_fen(to_fen(x)) still matches x, and to_fen(from_fen(fen)) is stable
    under any number of further round trips -- it is only the very first
    string that can be rewritten.
    """
    ranks = []
    for row in chess.chess_board:
        text = ""
        empty = 0
        for piece in row:
            if piece == 0:
                empty += 1
                continue
            #Runs of empty squares are written as their length
            if empty:
                text += str(empty)
                empty = 0
            letter = FEN_LETTERS[abs(piece)]
            text += letter if piece > 0 else letter.lower()
        if empty:
            text += str(empty)
        ranks.append(text)
    #Row 0 is rank 8, which is also the rank FEN starts from
    placement = "/".join(ranks)

    castling = ("K" if chess.white_king_castle else "") \
             + ("Q" if chess.white_queen_castle else "") \
             + ("k" if chess.black_king_castle else "") \
             + ("q" if chess.black_queen_castle else "")

    #en_passant maps {capturing pawn square: destination}, so every value is
    #the same square: the one the capturing pawn would land on
    if chess.en_passant:
        destination = next(iter(chess.en_passant.values()))
        ep_field = square_name(destination.row, destination.col)
    else:
        ep_field = "-"

    return " ".join((
        placement,
        "w" if chess.side_to_move > 0 else "b",
        castling or "-",
        ep_field,
        str(chess.half_move_clock),
        str(chess.full_move_count),
    ))

def parse_uci_move(uci):
    from_col = COLUMN_CODE[uci[0]]
    from_row = 8 - int(uci[1])
    to_col = COLUMN_CODE[uci[2]]
    to_row = 8 - int(uci[3])
    promotion = 0
    if len(uci) == 5:
        promotion = UCI_PROMOTION[uci[4]]

    return ((from_row, from_col), Move(to_row, to_col, promotion))