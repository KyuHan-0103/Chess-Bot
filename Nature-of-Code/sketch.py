"""The Nature of Code -- evolving a flock of vehicles with a genetic algorithm.

Nothing here knows how to steer. Each vehicle carries DNA: one random force
vector per frame of its lifespan, played back in order. Most of the first
generation wanders off, but the ones that drift nearest the target get the most
tickets in the mating pool, so their genes dominate the next generation. Over a
few dozen generations the flock evolves a flight path to the target.

Click anywhere on the canvas to drop a new target at the mouse position.

tkinter port of sketch.js. Run with:  python sketch.py
"""

import tkinter as tk

import numpy as np

from objects import LIFESPAN, Population, Obstacle

WIDTH, HEIGHT = 1000, 700
FRAME_MS = 16          # ~60 fps, p5's default frame rate
GREY = "#7f7f7f"       # p5's fill(127)
GREEN = "#4caf50"      # a vehicle that reached the target
BLACK = "#000000"      # p5's stroke(0)
STROKE = 2             # p5's strokeWeight(2)
TARGET_DIAMETER = 48
POPULATION_SIZE = 100
# Centred horizontally and lifted above the middle of the canvas
TARGET_START = (WIDTH / 2, HEIGHT / 4)


class Sketch:
    """Equivalent of setup() plus draw()."""

    def __init__(self, root):
        self.canvas = tk.Canvas(
            root,
            width=WIDTH,
            height=HEIGHT,
            bg="white",              # p5's background(255)
            highlightthickness=0,    # style.css: no border, no padding
        )
        self.canvas.pack()

        # The sketch owns the target and hands it to the population, which
        # hands it to each vehicle.
        self.target = np.array(TARGET_START, dtype=float)

        self.obstacle1_id = self.canvas.create_polygon(410, 460, 410, 430, 80, 220, 80, 600, fill=GREY, outline=BLACK)
        self.obstacle2_id = self.canvas.create_polygon(410, 360, 410, 330, 80, 120, 80, 150, fill=GREY, outline=BLACK)
        self.obstacle1 = Obstacle(410, 460, 410, 430, 80, 220, 80, 600)
        self.obstacle2 = Obstacle(410, 360, 410, 330, 80, 120, 80, 150)
        
        self.population = Population(POPULATION_SIZE, self.target, self.obstacle1, self.obstacle2, WIDTH, HEIGHT)

        # Canvas items are created once and repositioned each frame. tkinter's
        # canvas is retained, not cleared-and-redrawn like a p5 sketch, so
        # moving existing items avoids the flicker you'd get from calling
        # delete("all") sixty times a second.
        self.vehicle_ids = []
        for vehicle in self.population.vehicles:
            self.vehicle_ids.append(self.canvas.create_polygon(
                0, 0, 0, 0, 0, 0, fill=GREY, outline=BLACK, width=STROKE
            ))
        # Recolouring is the one thing worth tracking, since setting a fill that
        # is already set still dirties the item and forces a redraw.
        self.vehicle_fills = [GREY] * len(self.vehicle_ids)

        self.target_id = self.canvas.create_oval(
            0, 0, 0, 0, fill=GREY, outline=BLACK, width=STROKE
        )
        self.draw_target()

        self.status_id = self.canvas.create_text(
            12, 12, anchor="nw", fill=BLACK, font=("TkDefaultFont", 14), text=""
        )

        # p5's mousePressed(): a click moves the target under the pointer
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw()

    def on_click(self, event):
        """A new target appears wherever the mouse was clicked."""
        self.set_target(event.x, event.y)

    def set_target(self, x, y):
        self.target = np.array([float(x), float(y)])
        self.population.set_target(self.target)
        self.draw_target()

    def draw_target(self):
        """Draw an ellipse at the target position."""
        r = TARGET_DIAMETER / 2
        self.canvas.coords(
            self.target_id,
            self.target[0] - r, self.target[1] - r,
            self.target[0] + r, self.target[1] + r,
        )

    def draw(self):
        # Live out one frame of this generation, or breed the next one
        self.population.update()

        vehicles = self.population.vehicles
        for i, (vehicle, v_id) in enumerate(zip(vehicles, self.vehicle_ids)):
            self.canvas.coords(v_id, *vehicle.shape_points())

            fill = GREEN if vehicle.arrived else GREY
            if fill != self.vehicle_fills[i]:
                self.canvas.itemconfig(v_id, fill=fill)
                self.vehicle_fills[i] = fill

        self.canvas.itemconfig(self.status_id, text=(
            f"generation {self.population.generation}    "
            f"frame {self.population.frame}/{LIFESPAN}    "
            f"arrived {self.population.arrived_count()}/{len(vehicles)}    "
            f"closest {self.population.closest_distance():.0f}px"
        ))

        self.canvas.after(FRAME_MS, self.draw)


def main():
    root = tk.Tk()
    root.title("The Nature of Code -- Genetic Algorithm")
    root.geometry("1000x700")
    Sketch(root)
    root.mainloop()


if __name__ == "__main__":
    main()
