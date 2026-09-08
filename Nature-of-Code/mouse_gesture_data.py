import tkinter as tk
import pandas as pd
class gesture_capture():

    def __init__(self, window):
        self.initial_x = None
        self.initial_y = None
        self.window = window

        self.canvas = tk.Canvas(window, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.current_ID = None
        self.data = []
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_click(self, event):
        self.initial_x = event.x
        self.initial_y = event.y
        
    def on_drag(self, event):
        # 3. Clear the previous frame's line so it doesn't leave a trail
        if self.current_ID:
            self.canvas.delete(self.current_ID)
        
        # Draw a new line from the initial click to the current mouse position
        self.current_ID = self.canvas.create_line(
            self.initial_x, self.initial_y, event.x, event.y, 
            fill="blue", width=5
        )
    
    def on_release(self, event):
        if self.current_ID:
            self.canvas.delete(self.current_ID)
            self.current_ID = None
        
        delta_x = event.x - self.initial_x
        delta_y = event.y - self.initial_y
        label = self.determine_direction(delta_x, delta_y)
        self.add_data_point(delta_x, delta_y, label)

    def add_data_point(self, x, y, label):
        new_row = {
            "x": x,
            "y": y,
            "label": label
        }
        self.data.append(new_row)

    def determine_direction(self, delta_x, delta_y):
        if abs(delta_x) > abs(delta_y):
            if delta_x > 0: return "right"
            else: return "left"
        else:
            if delta_y > 0: return "down"
            else: return "up"
def main():
    canvas = tk.Tk()
    canvas.geometry("1000x800")
    capture = gesture_capture(canvas)

    # Keep the window running manually while data is less than 10
    while len(capture.data) < 10:
        try:
            # Manually process Tkinter events and redraw the window
            canvas.update_idletasks()
            canvas.update()
        except tk.TclError:
            # Handles the case where the user closes the window manually before reaching 10 rows
            break

    # Once the loop breaks (because length reached 10), destroy the window
    try:
        canvas.destroy()
    except tk.TclError:
        # Window was already manually closed
        pass

    df = pd.DataFrame(capture.data)
    df.to_csv('gesture.csv', index=False)

if __name__ == "__main__":
    main()

