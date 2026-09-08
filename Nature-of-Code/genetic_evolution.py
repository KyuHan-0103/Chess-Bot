"""
Purpose of this program is to "simulate" genetic evolution

Step 1: Create a population of N elements, each with randomly generated DNA (properties)

Step 2: Selection
    - Evaluate Fitness
        - to evaluate fitness, a function producing the "fitness" score is created
    - Create a mating pool
        - to choose which element "deserves" to be "bred" we could be elitist and take the best
        - or we could take the top 50%, but those are solutions made simply out of ease
        - instead we can make a wheel of "fortune." The chance each element has of being selected
        - to be reproduced is equal to its "fitness" score / total fitness of the entire population

Step 3: Reproduction
    - Crossover
        - Multiple options for this: could do 50/50 or just choose a random midpoint
        - could coin toss for each gene
    - Mutation
        - Mutation is described by rate. A given genetic alg may have a 0.1 percent chance of
        - mutation

Step 4: Repetition
    - Return to step two and repeat

Two genomes live here, both following those four steps:

    DNA        a sequence of random force vectors, one per frame of a vehicle's
               lifespan. This is the genome the flock in objects.py evolves.
    StringDNA  the original demo, guessing a word one character at a time.
               Run this file directly to watch it: python genetic_evolution.py
"""

import random
import string
import math

import numpy as np

MUTATION_RATE = 0.01

#One generator for the whole simulation, seeded from the OS
rng = np.random.default_rng()


class DNA:
    """A sequence of steering forces, one gene per frame of a lifespan.

    A gene is a random vector: a random heading scaled by a random magnitude
    no larger than the force its vehicle is able to apply. Play the genes back
    in order and you get the whole flight path, so the genome *is* the
    behaviour -- there is no steering logic left to write.
    """

    def __init__(self, lifespan, max_force, genes=None):
        self.lifespan = int(lifespan)
        self.max_force = float(max_force)

        #Step 1: random DNA, unless we were handed genes by a crossover
        if genes is None:
            genes = np.array([self.random_gene() for _ in range(self.lifespan)])
        self.genes = genes

    def random_gene(self):
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0, self.max_force)
        return np.array([np.cos(angle), np.sin(angle)]) * magnitude

    #Step 3: Crossover. A coin toss per gene, same as the string demo below
    def crossover(self, partner):
        mine = rng.random(self.lifespan) < 0.5
        #mine[:, None] broadcasts the per-gene choice across both x and y, so a
        #gene is inherited whole rather than split into one parent's x and the
        #other's y, which would be a direction neither parent ever had
        genes = np.where(mine[:, None], self.genes, partner.genes)
        return DNA(self.lifespan, self.max_force, genes)

    #Step 3: Mutation. Every gene gets an independent roll of the dice
    def mutate(self, rate=MUTATION_RATE):
        for i in range(self.lifespan):
            if rng.random() < rate:
                self.genes[i] = self.random_gene()


class StringDNA:
    """The original demo genome: one lowercase character per gene."""

    def __init__(self, target):
        self.genes = []
        self.length = len(target)
        self.target = target
        self.fitness = 0.01

        self.pool = string.ascii_lowercase + " "
        for i in range(self.length):
            self.genes.append(random.choice(self.pool))

    def fitness_evaluation(self):
        #"fitness" = number of correct characters
        score = 0
        for i in range(len(self.genes)):
            if self.genes[i] == self.target[i]:
                score += 1
        self.fitness = score / len(self.genes)

    def crossover(self, partner):
        child = StringDNA(self.target)
        for i in range(self.length):
            if random.random() < 0.5:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = partner.genes[i]
        return child

    def mutate(self):
        for i in range(len(self.genes)):
            num = random.random()
            if num < 0.01:
                self.genes[i] = random.choice(self.pool)


POPULATION = []
COUNT = 0
TARGET = list('cattle')

def setup():
    global TARGET
    for i in range(150):
        POPULATION.append(StringDNA(TARGET))

def draw():
    global COUNT
    global POPULATION

    #Get fitness score for each element
    for element in POPULATION:
        element.fitness_evaluation()
        guess = ''.join(element.genes)
        COUNT += 1
        print(f"{COUNT}: {guess}")
        if element.genes == element.target:
            print(f"{COUNT}: {guess} WINNER!")
            return True

    #Create mating pool
    mating_pool = []
    for e in POPULATION:
        n = math.floor(e.fitness * 100)

        for i in range(n):
            mating_pool.append(e)


    children = []
    for i in range(len(POPULATION)):
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        child = parent_A.crossover(parent_B)
        child.mutate()
        children.append(child)

    POPULATION[:] = children

    return False


def main():
    setup()
    while True:
        if draw():
            break

if __name__ == "__main__":
    main()
