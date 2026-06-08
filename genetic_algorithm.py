import random

from prototype_dev import Strategy_prey, Population_state, Position


# Generate n random prey.
def generate_prey(generations: int) -> list[Strategy_prey]:
    # Create list to store prey.
    prey: list[Strategy_prey] = []

    # Create n prey.
    for _ in range(generations):
        pos = random.choice([
            Position.LEADER,
            Position.FOLLOWER,
            Position.BORDER,
            Position.CENTER
        ])
        strat: Strategy_prey = generate_for_position(pos)
        prey.append(strat)

    # Return generated prey.
    return prey

def generate_for_position(pos: Position) -> Strategy_prey:
    strat = Strategy_prey(
        p_F_given_L=random.random(),
        p_L_given_F=random.random(),
        p_L_given_B=random.random(),
        position=pos
    )

    return strat

def create_generation(generation: list[tuple[Strategy_prey, float]], mu: float = 0.1, k: int = 4, sigma: float = 0.02) -> list[Strategy_prey]:
    new_generation: list[Strategy_prey] = []

    n = len(generation)

    for i in range(int(n / 2)):
        parent1 = select_parent(generation, k)

        parent2: Strategy_prey
        while True:
            parent2 = select_parent(generation, k)
            if parent1 != parent2:
                break

        child1, child2 = combine_genes(parent1, parent2, mu, sigma)
        new_generation.append(child1)
        new_generation.append(child2)

    return new_generation

def select_parent(generation: list[tuple[Strategy_prey, float]], k: int) -> Strategy_prey:
    candidates: list[tuple[Strategy_prey, float]] = []

    n = len(generation)

    for _ in range(k):
        index = random.randint(0, n - 1)
        candidates.append(generation[index])

    return max(candidates, key=lambda x: x[1])[0]

def combine_genes(parent1: Strategy_prey, parent2: Strategy_prey, mu: float, sigma: float) -> tuple[Strategy_prey, Strategy_prey]:
    split: int = random.choice([1,2])

    child1: Strategy_prey
    child2: Strategy_prey

    if split == 1:
        child1 = Strategy_prey(
            p_F_given_L=parent1.p_F_given_L,
            p_L_given_F=parent2.p_L_given_F,
            p_L_given_B=parent2.p_L_given_B,
            position=parent2.position # reassigned later
        )
        child2 = Strategy_prey(
            p_F_given_L=parent2.p_F_given_L,
            p_L_given_F=parent1.p_L_given_F,
            p_L_given_B=parent1.p_L_given_B,
            position=parent1.position  # reassigned later
        )

    else:
        child1 = Strategy_prey(
            p_F_given_L=parent1.p_F_given_L,
            p_L_given_F=parent1.p_L_given_F,
            p_L_given_B=parent2.p_L_given_B,
            position=parent2.position  # reassigned later
        )
        child2 = Strategy_prey(
            p_F_given_L=parent2.p_F_given_L,
            p_L_given_F=parent2.p_L_given_F,
            p_L_given_B=parent1.p_L_given_B,
            position=parent1.position  # reassigned later
        )

    return mutate(child1, mu, sigma), mutate(child2, mu, sigma)

def mutate(strategy: Strategy_prey, mu: float, sigma: float) -> Strategy_prey:
    p1 = random.random()
    if p1 < mu:
        strategy.p_F_given_L = strategy.p_F_given_L + random.gauss(0, sigma)
        if strategy.p_F_given_L < 0:
            strategy.p_F_given_L = 0
        elif strategy.p_F_given_L > 1:
            strategy.p_F_given_L = 1

    p2 = random.random()
    if p2 < mu:
        strategy.p_L_given_F = strategy.p_L_given_F + random.gauss(0, sigma)
        if strategy.p_L_given_F < 0:
            strategy.p_L_given_F = 0
        elif strategy.p_L_given_F > 1:
            strategy.p_L_given_F = 1

    p3 = random.random()
    if p3 < mu:
        strategy.p_L_given_B = strategy.p_L_given_B + random.gauss(0, sigma)
        if strategy.p_L_given_B < 0:
            strategy.p_L_given_B = 0
        elif strategy.p_L_given_B > 1:
            strategy.p_L_given_B = 1

    return strategy