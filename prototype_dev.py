# Evolutionary algorithm implementation 
# We have re-produced the results from Yang et. al.(2024) which will act as a baseline for our research. 

# Imports 
import random
from dataclasses import dataclass
from enum import Enum
import math

from position import Position
from position_history import PositionHistory


@dataclass
class Strategy_prey:
    p_F_given_L: float
    p_L_given_F: float
    p_L_given_B: float

    position: Position

    history: PositionHistory = PositionHistory()

    def validation(self) -> None:
        values = [self.p_F_given_L, self.p_L_given_F, self.p_L_given_B]
        for value in values:
            if not (0.0 <= value <= 1.0):
                raise ValueError("All probabilities must be between 0 and 1.")

    def update_position(self) -> None:
        self.validation()

        r = random.random()

        if self.position == Position.LEADER:
            if r < self.p_F_given_L:
                self.position = Position.FOLLOWER
            else:
                self.position =  Position.LEADER
            return

        elif self.position == Position.FOLLOWER:
            if r < self.p_L_given_F:
                self.position =  Position.LEADER
            else:
                self.position =  Position.FOLLOWER
            return

        elif self.position == Position.BORDER:
            if r < self.p_L_given_B:
                self.position =  Position.LEADER
            else:
                self.position =  Position.BORDER
            return

        elif self.position == Position.CENTER:
            self.position =  Position.CENTER
            return

        raise ValueError(f"Unknown position: {self.position}")
            

@dataclass
class Env:
    X_L: float
    X_F: float
    X_B: float
    X_C: float

    def normalize(self) -> "Env":
        total = self.X_L + self.X_F + self.X_B + self.X_C

        if total <= 0:
            raise ValueError("Check the Risk values. Total must be greater than 0.")
        
        return Env(
            X_L=self.X_L / total,
            X_F=self.X_F / total,
            X_B=self.X_B / total,
            X_C=self.X_C / total
        ) 
    
    def vector(self) -> list[float]:
        return [self.X_L, self.X_F, self.X_B, self.X_C]
    

@dataclass
class Population_state:
    n_L : int
    n_F : int
    n_B : int
    n_C : int

    def validation(self) -> None:
        values = [self.n_L, self.n_F, self.n_B, self.n_C]
        for value in values:
            if value < 0:
                raise ValueError("All population counts must be non-negative.")

        if any(not isinstance(v, int) for v in values):
            raise TypeError("Population counts must be integers.")

    @property
    def total(self) -> int:
        return self.n_L + self.n_F + self.n_B + self.n_C
    def vector(self) -> list[int]:
        return [self.n_L, self.n_F, self.n_B, self.n_C]

# def expand_state_to_positions(state: Population_state) -> list[Position]:
#     state.validation()
#
#     return (
#         [Position.LEADER] * state.n_L
#         + [Position.FOLLOWER] * state.n_F
#         + [Position.BORDER] * state.n_B
#         + [Position.CENTER] * state.n_C
#     )


def compress_positions_to_state(strategies: list[Strategy_prey]) -> Population_state:
    return Population_state(
        n_L=sum(s.position == Position.LEADER for s in strategies),
        n_F=sum(s.position == Position.FOLLOWER for s in strategies),
        n_B=sum(s.position == Position.BORDER for s in strategies),
        n_C=sum(s.position == Position.CENTER for s in strategies),
    )


def update_population_once(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> Population_state:

    for prey in strategies:
        prey.update_position()

    enforce_geometric_constraints(strategies, r_F_L=r_F_L)

    return compress_positions_to_state(strategies)

def enforce_mobile_group_constraint(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> None:
    """
    Ensures followers do not exceed the allowed follower/leader ratio.
    If there are too many followers, excess followers are moved to border.
    """
    state = compress_positions_to_state(strategies)

    if state.n_L == 0:
        for prey in strategies:
            if prey.position == Position.FOLLOWER:
                prey.position = Position.BORDER
        return

    max_followers = r_F_L * state.n_L

    if state.n_F <= max_followers:
        return

    excess_followers = state.n_F - max_followers

    followers = []
    for strategy in strategies:
        if strategy.position == Position.FOLLOWER:
            followers.append(strategy)

    while excess_followers > 0:
        excess_followers -= 1
        follower = random.choice(followers)
        follower.position = Position.BORDER
        followers.remove(follower)

def max_center_count(herd_size: int) -> int:
    """
    Approximates the maximum number of center individuals
    allowed for a cohesive herd of size H = B + C.
    """

    if herd_size <= 0:
        return 0

    value = (herd_size ** (1 / 3) - 2) #** 3

    return max(0, math.floor(value))


def enforce_cohesive_group_constraint(
    strategies: list[Strategy_prey],
) -> None:
    """
    Ensures the number of center individuals is geometrically plausible.
    If too many center individuals exist, move excess to border.
    """
    state = compress_positions_to_state(strategies)

    herd_size = state.n_B + state.n_C
    allowed_center = max_center_count(herd_size)

    if state.n_C <= allowed_center:
        return

    excess_center = state.n_C - allowed_center

    centers = []
    for strategy in strategies:
        if strategy.position == Position.CENTER:
            centers.append(strategy)

    while excess_center > 0:
        excess_center -= 1
        center = random.choice(centers)
        center.position = Position.BORDER
        centers.remove(center)

def enforce_geometric_constraints(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> None:
    enforce_mobile_group_constraint(strategies, r_F_L=r_F_L)
    enforce_cohesive_group_constraint(strategies)

def random_relocation(
    state: Population_state,
    r_F_L: int = 8,
) -> Population_state:
    positions = [
        Position.LEADER,
        Position.FOLLOWER,
        Position.BORDER,
        Position.CENTER,
    ]

    all_positions = random.choices(positions, k=state.total)

    new_state = compress_positions_to_state(all_positions)
    new_state = enforce_geometric_constraints(new_state, r_F_L=r_F_L)
    new_state.validation()
    return new_state

def run_simulation(
    initial_state: Population_state,
    strategy: Strategy_prey,
    steps: int = 40000, #initial 500
    burn_in: int = 1000, #initial 100
    r_F_L: int = 8,
    relocation_interval: int = 200,
) -> dict[str, float]:
    """
    Runs the behavioural simulation and returns long-run positional frequencies.
    burn_in steps are ignored so early transient behaviour does not dominate.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if burn_in < 0 or burn_in >= steps:
        raise ValueError("burn_in must be >= 0 and smaller than steps.")

    state = initial_state
    state.validation()
    strategy.validation()

    counts = {
        "L": 0,
        "F": 0,
        "B": 0,
        "C": 0,
    }

    measured_steps = 0

    for t in range(steps):
        state = update_population_once(
            state=state,
            strategy=strategy,
            r_F_L=r_F_L,
        )

        state = random_relocation(state, r_F_L=r_F_L)

        if (t + 1) % relocation_interval == 0:
            state = random_relocation(state)

        if t >= burn_in:
            counts["L"] += state.n_L
            counts["F"] += state.n_F
            counts["B"] += state.n_B
            counts["C"] += state.n_C
            measured_steps += 1

    total_observations = initial_state.total * measured_steps

    return {
        "L": counts["L"] / total_observations,
        "F": counts["F"] / total_observations,
        "B": counts["B"] / total_observations,
        "C": counts["C"] / total_observations,
    }

def compute_risk(
    frequencies: dict[str, float],
    environment: Env,
) -> float:
    """
    Computes Yang-style positional risk: X^T f.
    Lower risk is better for prey.
    """

    env = environment.normalize()

    return (
        env.X_L * frequencies["L"]
        + env.X_F * frequencies["F"]
        + env.X_B * frequencies["B"]
        + env.X_C * frequencies["C"]
    )


def compute_prey_fitness(
    frequencies: dict[str, float],
    environment: Env,
) -> float:
    """
    Converts risk into fitness.
    Higher fitness is better.
    """

    risk = compute_risk(frequencies, environment)
    return 1.0 - risk

def evaluate_strategy(
    strategy: Strategy_prey,
    initial_state: Population_state,
    environment: Env,
    steps: int = 40000, #initial 500
    burn_in: int = 1000, #initial 100
    r_F_L: int = 8,
    relocation_interval: int = 200,
) -> tuple[dict[str, float], float]:
    """
    Runs simulation and returns:
    1. positional frequencies f
    2. prey fitness score
    """

    frequencies = run_simulation(
        initial_state=initial_state,
        strategy=strategy,
        steps=steps,
        burn_in=burn_in,
        r_F_L=r_F_L,
        relocation_interval=relocation_interval,
    )

    fitness = compute_prey_fitness(
        frequencies=frequencies,
        environment=environment,
    )

    return frequencies, fitness