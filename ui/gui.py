import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from core.constants import PIECE_NAMES, MATE
from core.movegen import *
from core.rules import *
from engine.search import *

PIECE_ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "piece_icons"

PIECE_SCALE = 0.9
LIGHT_SQUARE = "#ebd6b0"
DARK_SQUARE = "#a97e5c"
PICKED_UP_SQUARE = "#d8c65a"
DEPTH = 8
TIME_LIMIT = 8.0
#How far the pointer has to move before a press counts as a drag instead of a click
DRAG_THRESHOLD = 4

STATE = {
    "chess": None,      #Chess object holding the matrix
    "cells": {},        #(row, col) in the square frame
    "pieces": {},       #(row, col) canvas holding the piece
    "selected": None,   #(row, col) of the piece that is currently picked up 
    "legal_moves": {},  #(row, col) -> list of Moves that originate from location
    "origin": None,     #(row, col) where the piece originally started
    "press": None,      #Initial coords of the press, differentiating a click from a drag
    "floater": None     #Labl that follows the cursor while dragging
}
#Keeps frame object square sized (Only triggered on main window size changing)
def make_square(event, target_frame, window):
    #Ensures the event came from the window widget
    if event.widget == window:
        #Finds smallest dimension of the window
        size = min(window.winfo_width(), window.winfo_height())
        target_frame.config(width= size, height= size)

def fit_image(event, canvas):

    #Find largest icon size that still fits into cell
    size = max(1, int(min(event.width, event.height) * PIECE_SCALE))

    #Only rebuild image if the size changed
    if size != canvas.piece_size:
        canvas.piece_size = size
        resized = canvas.pil_image.resize((size, size), Image.LANCZOS)
        #Store the PhotoImage so that Python doesn't dump it

        canvas.image = ImageTk.PhotoImage(resized)
        canvas.itemconfig(canvas.piece_item, image= canvas.image)

    #Recenter's the icon in the cell
    canvas.coords(canvas.piece_item, event.width // 2, event.height // 2)

#Returns the color a square should be when nothing is picked up
def square_color(ROW, COL):
    return LIGHT_SQUARE if (ROW + COL)%2 == 0 else DARK_SQUARE

#Which square a set of screen coordinates falls on, or None if it is off the board
#This works off root coordinates on purpose, winfo_containing would just return the
#floating drag image because that sits directly under the pointer
def identify_cell(x_root, y_root):
    for position, cell in STATE["cells"].items():
        left = cell.winfo_rootx()
        top = cell.winfo_rooty()
        if left <= x_root < left + cell.winfo_width():
            if top <= y_root < top + cell.winfo_height():
                return position

    return None

#Highlights a square either as picked up or back to its normal color
def highlight(position, picked_up):
    if position not in STATE["cells"]:
        return
    color = PICKED_UP_SQUARE if picked_up else square_color(*position)
    STATE["cells"][position].config(bg=color)
    piece = STATE["pieces"].get(position)
    if piece is not None:
        piece.config(bg= color)

#Adds chosen position to selected state and highlights the position
def pick_up(position):
    STATE["selected"] = position
    STATE["legal_moves"] = moves_by_square(position)
    highlight(position, True)

#Puts whatever is picked up back down, which is also how a piece "returns"
#after an invalid click or drop
def put_down():
    if STATE["selected"] is not None:
        highlight(STATE["selected"], False)
        STATE["selected"] = None
        STATE["legal_moves"] = {}

#A target is only usable if it is on the board, is not the square the piece
#started on, and is not occupied by one of your own pieces
#Note this does not consider king move restrictions
def is_legal_target(origin, target):
    if target is None or target == origin:
        return False
    
    return target in STATE["legal_moves"]

#Removes piece at given position
#If a piece existed there, destroys said piece
def clear_square(position):
    piece = STATE["pieces"].pop(position, None)
    if piece is not None:
        piece.destroy()

#Relaods board from chess.chess_board
#Castling and En passant affect move that just the origin and target squares
#So patching individual squares desync the window and the board
def refresh_board():
    for position in list(STATE["pieces"]):
        clear_square(position)

    matrix = STATE["chess"].chess_board
    for r, row in enumerate(matrix):
        for c, square in enumerate(row):
            if square != 0:
                place_piece(r, c, square, STATE["cells"][(r, c)])
        
def move_piece(origin, target):
    candidates = STATE["legal_moves"].get(target, [])
    put_down()
    if not candidates:
        return

    if len(candidates) == 1:
        move = candidates[0]
    else:
        #More than one move on the same square means promotion
        wanted = ask_promotion()
        if wanted is None:
            return
        move = next(m for m in candidates if m.promotion == wanted)

    #If move is illegal (leaves king in check) render it invalid
    if not make_move(STATE["chess"], origin, move):
        print("Illegal Move", origin, "->", target)
        refresh_board()
        return

    #Render AFTER board has been updated
    refresh_board()

#Groups this piece's legal moves by destination square, because four moves can
#share a square when a pawn promotes and a click only identifies the square.
#available_moves is pseudo-legal for everything except the king, so trying each
#move is also what makes the highlighting honest for pinned pieces.
def moves_by_square(position):
    matrix = STATE["chess"]
    table = {}
    for move in available_moves(matrix, *position):
        if make_move(matrix, position, move):
            undo_move(matrix)
            table.setdefault((move.row, move.col), []).append(move)
    return table

#Handles plain clicks, on a piece or empty square
def on_square_click(ROW, COL):
    matrix = STATE["chess"].chess_board
    selected = STATE["selected"]
    
    #If nothing is picked up yet, pick up the piece on this square
    #A destination square is empty or holds an enemy, so testing it up front
    #blocks every move you could possibly make.
    if selected is None:
        if side_of(matrix[ROW][COL]) == STATE["chess"].side_to_move:
                pick_up((ROW, COL))
        return

    #Clicking the same piece puts it down
    if selected == (ROW, COL):
        put_down()
        return

    if is_legal_target(selected, (ROW, COL)):
        move_piece(selected, (ROW, COL))
    else:
        #Invalid square so the piece goes back to where it was
        #Clicking one of your own pieces picks up that piece instead
        put_down()
        if side_of(matrix[ROW][COL]) == STATE["chess"].side_to_move:
            pick_up((ROW, COL))

#Clicking anywhere that is not a square cancels the pick up.
#Bound on the toplevel, which sits in every child widget's bindtags, so this
#runs after the square's own handler has already had its turn.
def on_window_click(event):
    if identify_cell(event.x_root, event.y_root) is None:
        put_down()

def on_press(event, ROW, COL):
    STATE["origin"] = (ROW, COL)
    STATE["press"] = (event.x_root, event.y_root)

def on_motion(event):
    origin = STATE["origin"]
    if origin is None:
        return
    
    #Wait until the pointer has actually moved before starting a drag,
    #otherwise every click would flicker a floating image
    if STATE["floater"] is None:
        start_x, start_y = STATE["press"]
        if abs(event.x_root - start_x) < DRAG_THRESHOLD:
            if abs(event.y_root - start_y) < DRAG_THRESHOLD:
                return
        start_drag(origin)

        if STATE["floater"] is None:
            return
    move_floater(event.x_root, event.y_root)

def on_release(event):
    origin = STATE["origin"]
    STATE["origin"] = None
    if origin is None:
        return

    #If the cursor wasn't dragged and only was a click
    if STATE["floater"] is None:
        on_square_click(*origin)
        return

    target = identify_cell(event.x_root, event.y_root)
    end_drag()

    #An invalid drop leaves the matrix alone, so unhiding the piece in end_drag
    #is what sends it back to the square it came from
    if is_legal_target(origin, target):
        move_piece(origin, target)
    else:
        put_down()
 
def start_drag(origin):
    matrix = STATE["chess"].chess_board

    #Checks if piece exists
    piece = STATE["pieces"].get(origin)
    if piece is None:
        return
    #Clears any previous click based pick up and starts a fresh one
    put_down()
    row, col = origin
    if side_of(matrix[row][col]) != STATE["chess"].side_to_move:
        return
    pick_up(origin)

    #Hides the real piece so it is not drawn in two places at once
    piece.itemconfig(piece.piece_item, state="hidden")

    #A canvas item cannot be dragged outside its own canvas, so creates
    #a new label which follows the cursor
    top = piece.winfo_toplevel()
    floater = tk.Label(top, image= piece.image, bd= 0, highlightthickness= 0, bg= PICKED_UP_SQUARE)
    floater.image = piece.image
    floater.lift()
    STATE["floater"] = floater

def move_floater(x_root, y_root):
    floater = STATE["floater"]
    top = floater.winfo_toplevel()

    #place() wants coordinates relative to the toplevel, and anchor keeps the
    #icon centred on the pointer
    floater.place(x= x_root - top.winfo_rootx(), y= y_root - top.winfo_rooty(), anchor="center")

def end_drag():
    floater = STATE["floater"]
    STATE["floater"] = None

    if floater is not None:
        floater.destroy()

    #Shows the real piece again where ever it ends up
    piece = STATE["pieces"].get(STATE["selected"])
    if piece is not None:
        piece.itemconfig(piece.piece_item, state="normal")

def ask_promotion():
    parent = STATE["cells"][(0, 0)].winfo_toplevel()
    top = tk.Toplevel(parent)
    top.title("Promote to")
    top.resizable(False, False)
    top.transient(parent)
    top.grab_set()

    chosen = {"piece": None}

    def pick(piece_type):
        chosen["piece"] = piece_type
        top.destroy()

    for column, piece_type in enumerate(PROMOTION_PIECES):
        tk.Button(top, text=PIECE_NAMES[piece_type].capitalize(), width=8,
                  command=lambda p=piece_type: pick(p)).grid(row=0, column=column, padx=4, pady= 8)

    top.wait_window()
    return chosen["piece"]

def place_piece(ROW, COL, piece, cell):

    piece_name = PIECE_NAMES[abs(piece)]
    color = "white" if piece > 0 else "black"
    
    image_path = PIECE_ICON_DIR / (color + "_" + piece_name + ".png")

    #Place canvas inside the proper cell and grid it to have it fill the cell
    place_canvas = tk.Canvas(cell, bg= square_color(ROW, COL), highlightthickness=0)
    place_canvas.grid(row=0, column=0, sticky='nsew')


    place_canvas.pil_image = Image.open(image_path).convert('RGBA')
    place_canvas.piece_size = 0
    place_canvas.image = ImageTk.PhotoImage(place_canvas.pil_image)

    place_canvas.piece_item = place_canvas.create_image(0, 0, image= place_canvas.image)

    place_canvas.bind("<Configure>", lambda event: fit_image(event, place_canvas))

    #Press, motion and release are split so the same gesture can serve both a
    #click to pick up and a drag. Tk grabs the pointer on press, so motion and
    #release keep firing on this canvas even once the cursor has left it.
    place_canvas.bind("<ButtonPress-1>", lambda event: on_press(event, ROW, COL))
    place_canvas.bind("<B1-Motion>", on_motion)
    place_canvas.bind("<ButtonRelease-1>", on_release)

    STATE["pieces"][(ROW, COL)] = place_canvas

#Initiates Chess Board
def initiate_board(square_board_frame, chess_matrix) -> None:

    SQUARE_SIZE = 60

    #Resize rows and columns depending on window size
    for n in range(8):
            square_board_frame.grid_rowconfigure(n, weight= 1)
            square_board_frame.grid_columnconfigure(n, weight= 1)

    STATE["chess"] = chess_matrix
    #Creates squares and place pieces
    for r, row in enumerate(chess_matrix.chess_board):
        for c, square in enumerate(row):
            #Creates a frame and places it in the grid within the main frame
            cell = tk.Frame(square_board_frame, width = SQUARE_SIZE, height= SQUARE_SIZE, bg=square_color(r, c))
            cell.grid(row= r, column= c, sticky="nsew")

            #Lets the piece canvas stretch to fill the whole square
            cell.grid_rowconfigure(0, weight= 1)
            cell.grid_columnconfigure(0, weight= 1)

            #Stops child widgets from changing the frame size
            cell.grid_propagate(False)

            #Empty squares need to be clickable so that pieces can be placed on them
            cell.bind("<Button-1>", lambda event, ROW=r, COL=c: on_square_click(ROW, COL))

            STATE["cells"][(r, c)] = cell
            if square != 0:
                place_piece(r, c, square, cell)

def cpu_move():
    move, value = search(STATE["chess"], DEPTH, time_limit=TIME_LIMIT)
    if move is None:
        print("game over:", game_result(STATE["chess"]) or "no legal moves" )
        return
    make_move(STATE["chess"], *move)
    refresh_board()

    if value > MATE - 1000:
        print("cpu played", move, "--mate found")
    else:
        print("cpu played", move, "value", round(value, 2))

def play_game(chess):
    #Initialize Window
    window = tk.Tk()
    window.title('Chess Board')
    window.geometry("480x480")

    #Sets up central frame which will be resized in order to maintain square boardsize
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    square_board_canvas = tk.Canvas(window, width=480, height=480)
    square_board_canvas.grid(row= 0, column= 0)

    #Stops child widgets from changing frame size
    square_board_canvas.grid_propagate(False)

    initiate_board(square_board_canvas, chess)

    #Creates on event of when window size changes
    #On such an event the board size is changed to create a square
    window.bind("<Configure>", lambda event: make_square(event, square_board_canvas, window))

    #Catches clicks that land outside the board
    window.bind("<Button-1>", on_window_click)
    window.bind("<space>", lambda event: cpu_move())
    
    window.mainloop()