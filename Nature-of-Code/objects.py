import numpy as np
import math
from genetic_evolution import DNA, MUTATION_RATE, rng

#How many frames one generation gets to reach the target. A vehicle carries
#exactly this many genes, so its lifespan and its DNA are the same length.
LIFESPAN = 300


#An obstacle is a closed polygon, stored as its own coordinates in the same
#flat [x0, y0, x1, y1, ...] order tkinter's create_polygon takes. Keeping the
#geometry here means collision is exact math on the points, independent of
#whatever the canvas happens to be drawing.
class Obstacle:
    def __init__(self, *coords):
        #Accepts either Obstacle(410, 700, 410, 430, ...) or a single list of
        #the same numbers, so it can share a call with create_polygon
        if len(coords) == 1:
            coords = coords[0]
        self.points = np.asarray(coords, dtype=float).reshape(-1, 2)

    #Each vertex paired with the one after it, wrapping the last back to the
    #first so the polygon closes
    def edges(self):
        return self.points, np.roll(self.points, -1, axis=0)

    #True if the point is inside the polygon, or within margin of its outline.
    #A margin of the vehicle's radius is what turns "inside" into "touching".
    def contains(self, point, margin=0.0):
        px, py = float(point[0]), float(point[1])
        a, b = self.edges()

        #Ray casting: fire a ray to the right of the point and count the edges
        #it crosses. Odd means inside, even means outside.
        straddles = (a[:, 1] > py) != (b[:, 1] > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            #Where along each edge the ray's y sits, and the x it crosses at
            t = (py - a[:, 1]) / (b[:, 1] - a[:, 1])
            crossing_x = a[:, 0] + t * (b[:, 0] - a[:, 0])
        if np.count_nonzero(straddles & (px < crossing_x)) % 2 == 1:
            return True

        #Outside, but a body of some size can still be up against an edge
        return margin > 0.0 and self.distance(point) <= margin

    #Shortest distance from a point to the polygon's outline
    def distance(self, point):
        p = np.asarray(point, dtype=float)
        a, b = self.edges()

        edge = b - a
        length_sq = np.sum(edge * edge, axis=1)
        #How far along each edge its closest point sits, clamped to the ends so
        #a point off the side of an edge measures to the nearer vertex instead.
        #Zero-length edges divide by 1 and clamp to t=0, i.e. the vertex itself.
        t = np.sum((p - a) * edge, axis=1) / np.where(length_sq == 0.0, 1.0, length_sq)
        closest = a + np.clip(t, 0.0, 1.0)[:, None] * edge
        return float(np.min(np.linalg.norm(p - closest, axis=1)))


class Vehicle:
    def __init__(self, x, y, target, obstacles, width=1000, height=700, dna=None):
        #Every generation restarts from the same launch point, so the only
        #thing separating a good run from a bad one is the DNA
        self.start = np.array([x, y], dtype=float)
        self.position = self.start.copy()
        self.velocity = np.array([0, 0], dtype=float)
        self.acceleration = np.array([0, 0], dtype=float)

        self.radius_size = 8.0

        self.max_speed = 8.0
        self.max_force = 0.4

        #Canvas size, kept for reference. Nothing confines a vehicle to it:
        #it is free to fly off the window and steer its way back.
        self.width = float(width)
        self.height = float(height)

        #Step 1: a randomly generated genome, unless we were bred from parents
        self.dna = dna if dna is not None else DNA(LIFESPAN, self.max_force)

        self.fitness = 0.0
        self.arrived = False
        self.arrival_frame = None
        self.hit_obstacle = False
        #The point this vehicle is trying to reach, handed down by the population
        self.set_target(target)

        self.obstacles = obstacles

    #Copy the target so every vehicle keeps its own array and a later click
    #can't mutate it out from under the fitness math mid-frame
    def set_target(self, target):
        self.target = np.array(target, dtype=float)
        #A vehicle parked on the old target has not reached the new one, so it
        #goes back to flying its remaining genes rather than sitting there
        self.arrived = False
        self.arrival_frame = None

    def update(self, frame):
        #A vehicle that has already made it just sits on the target
        if self.hit_obstacle or self.arrived or frame >= self.dna.lifespan:
            return

        #The gene for this frame *is* the steering force for this frame
        self.apply_force(self.dna.genes[frame])

        #Update velocity
        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, self.max_speed)
        #Update pos
        self.position += self.velocity

        self.acceleration *= 0

        if np.linalg.norm(self.target - self.position) < self.radius_size * 2:
            self.arrived = True
            self.arrival_frame = frame

        self.check_obstacles()

    #A run is over the moment the vehicle's body meets an obstacle. The radius
    #is the margin, so grazing an edge counts as a hit and not just flying
    #dead centre into one.
    def check_obstacles(self):
        for obstacle in self.obstacles:
            if obstacle.contains(self.position, self.radius_size):
                self.hit_obstacle = True
                return

    #Step 2: how close did this genome get? Nearer is better, and actually
    #arriving is worth far more than merely hovering nearby.
    def evaluate_fitness(self):
        distance = np.linalg.norm(self.target - self.position)
        arrival_time = self.dna.lifespan
        if self.arrived:
            arrival_time = max(self.arrival_frame, 1)
        #+1 keeps a vehicle sitting exactly on the target from dividing by zero
        self.fitness = 1.0 / (arrival_time + distance)

        self.fitness = math.pow(self.fitness, 2)
        if self.hit_obstacle:
            self.fitness *= 0.1

        if self.arrived:
            self.fitness *= 4
        return self.fitness

    #Limit a vector
    def limit(self, curr_vector, limit_vector):
        cur_speed = np.linalg.norm(curr_vector)
        if cur_speed > limit_vector:
            curr_vector = (curr_vector/cur_speed) * limit_vector
        return curr_vector

    def apply_force(self, force):
        self.acceleration += force

    # A triangle pointing in the direction of velocity, in world coordinates.
    # Returns a flat [x0, y0, x1, y1, x2, y2] list for Canvas.coords().
    def shape_points(self):
        angle = np.arctan2(self.velocity[1], self.velocity[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        r = self.radius_size

        local = ((2 * r, 0.0), (-2 * r, -r), (-2 * r, r))
        points = []
        for x, y in local:
            points.append(self.position[0] + x * cos_a - y * sin_a)
            points.append(self.position[1] + x * sin_a + y * cos_a)
        return points


#Owns the flock and runs the genetic algorithm over it. It is also the single
#place the target lives between the sketch and the individual vehicles:
#Sketch -> Population -> Vehicle.
class Population:
    def __init__(self, count, target, obstacle1, obstacle2, width=800, height=600,
                 lifespan=LIFESPAN, mutation_rate=MUTATION_RATE):
        self.width = float(width)
        self.height = float(height)
        self.target = np.array(target, dtype=float)

        self.obstacles = [obstacle1, obstacle2]
        self.lifespan = lifespan
        self.mutation_rate = mutation_rate
        self.generation = 1
        self.frame = 0

        #The whole flock launches from the bottom middle of the canvas
        self.start = np.array([self.width / 2, self.height - 40], dtype=float)

        self.vehicles = []
        for i in range(count):
            self.vehicles.append(
                Vehicle(self.start[0], self.start[1],
                        self.target, self.obstacles, self.width, self.height))

    #Push a new target down to every vehicle
    def set_target(self, target):
        self.target = np.array(target, dtype=float)
        for vehicle in self.vehicles:
            vehicle.set_target(self.target)

    def update(self):
        #Step 4: the generation has lived out its lifespan, so breed the next one
        if self.frame >= self.lifespan:
            self.evolve()
            return

        for vehicle in self.vehicles:
            vehicle.update(self.frame)
        self.frame += 1

    #Step 2: Selection. A wheel of fortune, built by giving each vehicle a
    #number of tickets proportional to its fitness, so a vehicle twice as fit
    #as another is twice as likely to be picked as a parent.
    def mating_pool(self):
        best = max(vehicle.evaluate_fitness() for vehicle in self.vehicles)

        pool = []
        for vehicle in self.vehicles:
            #Scaled against the best of the generation, so the fittest vehicle
            #always gets the full 100 tickets no matter how far off it landed
            vehicle.fitness /= best
            for i in range(int(vehicle.fitness * 100)):
                pool.append(vehicle)
        return pool

    #Step 3: Reproduction, then back to a fresh run from the launch point
    def evolve(self):
        pool = self.mating_pool()

        children = []
        for i in range(len(self.vehicles)):
            parent_a = pool[rng.integers(len(pool))]
            parent_b = pool[rng.integers(len(pool))]
            child_dna = parent_a.dna.crossover(parent_b.dna)
            child_dna.mutate(self.mutation_rate)
            children.append(
                Vehicle(self.start[0], self.start[1], self.target,
                        self.obstacles, self.width, self.height, child_dna))

        self.vehicles = children
        self.frame = 0
        self.generation += 1

    #How the generation currently on screen is doing, for the sketch to display
    def arrived_count(self):
        return sum(1 for vehicle in self.vehicles if vehicle.arrived)

    def closest_distance(self):
        return min(float(np.linalg.norm(self.target - vehicle.position))
                   for vehicle in self.vehicles)
