import random

from prototype_dev import Strategy_prey, Population_state, Position


# Generate n random prey.
def generate_prey(state: Population_state) -> list[Strategy_prey]:
    # Create list to store prey.
    prey: list[Strategy_prey] = []

    # Create n prey.
    for i in range(state.n_L):
        strat: Strategy_prey = generate_for_position(Position.LEADER)
        prey.append(strat)

    for i in range(state.n_F):
        strat: Strategy_prey = generate_for_position(Position.FOLLOWER)
        prey.append(strat)

    for i in range(state.n_B):
        strat: Strategy_prey = generate_for_position(Position.BORDER)
        prey.append(strat)

    for i in range(state.n_C):
        strat: Strategy_prey = generate_for_position(Position.CENTER)
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

def compute_generation_fitness(prey: list[Strategy_prey]):

    return 0

# def create_generation(current_gen: list[Strategy_prey], mu: float = 0.1, k: int = 5):
#     new_generation: list[Strategy_prey] = []
#
#     n = len(current_gen)
#
#     for i in range(n):
#
#
#     return 0
#
# def select_parent(generation: list[Strategy_prey], k: int) -> Strategy_prey:
#     candidates: list[Strategy_prey] = []
#
#     n = len(generation)
#
#     for _ in range(k):
#         index = random.randint(0, n - 1)
#         candidates.append(generation[index])
#
#     # Needs to be fitness based!!
#
#     return parents
#
# def combine_genes(generation: list[Strategy_prey], mu: float, k: int) -> Strategy_prey:
