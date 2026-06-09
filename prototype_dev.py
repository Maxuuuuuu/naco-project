# Evolutionary algorithm implementation 
# We have re-produced the results from Yang et. al.(2024) which will act as a baseline for our research. 

# Imports 
import random
from dataclasses import dataclass
import math

from position import Position
from position_history import PositionHistory
from predator import Predator

class Strategy_prey:
    def __init__(self,  p_F_given_L: float, p_L_given_F: float, p_L_given_B: float, position: Position):
        self.p_F_given_L: float = p_F_given_L
        self.p_L_given_F: float = p_L_given_F
        self.p_L_given_B: float = p_L_given_B
        self.position: Position = position
        self.history: PositionHistory = PositionHistory()

    def copy(self):
        c = Strategy_prey(
            p_F_given_L=self.p_F_given_L,
            p_L_given_F=self.p_L_given_F,
            p_L_given_B=self.p_L_given_B,
            position=self.position,
        )
        return c

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
        prey.history.add(prey.position)
        prey.update_position()

    #a = count_positions(strategies)
    enforce_geometric_constraints(strategies, r_F_L=r_F_L)
    #b = count_positions(strategies)

    return compress_positions_to_state(strategies)

def count_positions(population: list[Strategy_prey]) -> dict[Position, int]:
    """
    Counts how many prey are currently in each position.
    """

    counts = {
        Position.LEADER: 0,
        Position.FOLLOWER: 0,
        Position.BORDER: 0,
        Position.CENTER: 0,
    }

    for prey in population:
        counts[prey.position] += 1

    return counts

def enforce_mobile_group_constraint(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> None:
    """
    Ensures followers do not exceed the allowed follower/leader ratio.
    If there are too many followers, excess followers are moved to border.
    """
    counts = count_positions(strategies)

    n_leaders = counts[Position.LEADER]
    n_followers = counts[Position.FOLLOWER]

    max_followers = r_F_L * n_leaders

    if n_followers <= max_followers:
        return

    excess_followers = n_followers - max_followers

    follower_agents = [
        prey for prey in strategies
        if prey.position == Position.FOLLOWER
    ]

    random.shuffle(follower_agents)

    for prey in follower_agents[:excess_followers]:
        prey.position = Position.BORDER

def max_center_count(herd_size: int) -> int:
    """
    Approximates the maximum number of center individuals
    allowed for a cohesive herd of size H = B + C.
    """

    if herd_size <= 0:
        return 0

    value = (herd_size ** (1 / 3) - 2) #** 3

    return max(0, math.floor(value))

def calculate_target_centres(herd_size: int) -> int:
    if herd_size < 8:
        return 0

    target_centres = int((herd_size ** (1 / 3) - 2) ** 3)

    return max(0, min(target_centres, herd_size))

def enforce_cohesive_group_constraint(
    strategies: list[Strategy_prey],
) -> None:
    """
    Ensures the number of center individuals is geometrically plausible.
    If too many center individuals exist, move excess to border.
    """
    herd_agents = [
        prey for prey in strategies
        if prey.position in {Position.BORDER, Position.CENTER}
    ]

    herd_size = len(herd_agents)
    target_centres = calculate_target_centres(herd_size)

    random.shuffle(herd_agents)

    for i, prey in enumerate(herd_agents):
        if i < target_centres:
            prey.position = Position.CENTER
        else:
            prey.position = Position.BORDER


def enforce_geometric_constraints(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> None:
    enforce_mobile_group_constraint(strategies, r_F_L=r_F_L)
    enforce_cohesive_group_constraint(strategies)

def random_relocation(
    strategies: list[Strategy_prey],
    r_F_L: int = 8,
) -> Population_state:
    positions = [
        Position.LEADER,
        Position.FOLLOWER,
        Position.BORDER,
        Position.CENTER,
    ]

    all_positions = random.choices(positions, k=len(strategies))

    for pos, strategy in zip(all_positions, strategies):
        strategy.position = pos

    enforce_geometric_constraints(strategies, r_F_L=r_F_L)
    return compress_positions_to_state(strategies)

def get_state_counts(strats: list[Strategy_prey]):
    lc = fc = bc = cc = 0

    for strat in strats:
        if strat.position == "L":
            lc += 1
        elif strat.position == "F":
            fc += 1
        elif strat.position == "B":
            bc += 1
        elif strat.position == "C":
            cc += 1

    state: Population_state = Population_state(lc, fc, bc, cc)

    return state

def run_simulation(
    initial_state: Population_state,
    strategies: list[Strategy_prey],
    steps: int = 500, #initial 500
    r_F_L: int = 8,
    relocation_interval: int = 200,
    use_coevolution: bool = False,
) -> dict[str, float]:
    """
    Runs the behavioural simulation and returns long-run positional frequencies.
    burn_in steps are ignored so early transient behaviour does not dominate.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")

    state = initial_state
    state.validation()
    for strategy in strategies:
        strategy.validation()

    counts = {
        "L": 0,
        "F": 0,
        "B": 0,
        "C": 0,
    }

    measured_steps = 0

    for t in range(steps):
        update_population_once(
            strategies=strategies,
            r_F_L=r_F_L,
        )

        #state = random_relocation(strategies, r_F_L=r_F_L)

        state = get_state_counts(strategies)

        if (t + 1) % relocation_interval == 0:
            state = random_relocation(strategies, r_F_L=r_F_L)

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
    strategy: Strategy_prey,
    environment: Env,
) -> float:
    """
    Computes Yang-style positional risk: X^T f.
    Lower risk is better for prey.
    """

    env = environment.normalize()

    # Total positions had.
    total: int = strategy.history.total()

    return 1.0 / (1.0 +(
        env.X_L * strategy.history.L_count
        + env.X_F * strategy.history.F_count
        + env.X_B * strategy.history.B_count
        + env.X_C * strategy.history.C_count
    ))


def compute_prey_fitness(
    strategies: list[Strategy_prey],
    environment: Env,
) -> list[tuple[Strategy_prey, float]]:
    """
    Converts risk into fitness.
    Higher fitness is better.
    """

    fitness: list[tuple[Strategy_prey, float]] = []

    for strategy in strategies:
        fit = compute_risk(strategy, environment)
        fitness.append((strategy, fit))

    return fitness

def holling_type_II(spent, total):
    h = total / 4 # Divide by possible positions.
    return spent / (spent + h)

def compute_pred_prey_fitness(
    strategies: list[Strategy_prey],
    predators: list[Predator],
):
    tL, tF, tB, tC = 0, 0, 0, 0

    for prey in strategies:
        tL += prey.history.L_count
        tF += prey.history.F_count
        tB += prey.history.B_count
        tC += prey.history.C_count

    total_prey_time = tL + tF + tB + tC

    pred_fitness: list[tuple[Predator, float]] = []

    preds = len(predators)
    pL, pF, pB, pC = 0.0, 0.0, 0.0, 0.0
    for predator in predators:
        pL += predator.L_predation
        pF += predator.F_predation
        pB += predator.B_predation
        pC += predator.C_predation

        # Compute predation applied. No need to normalize.
        fitness = (
                holling_type_II(tL, total_prey_time) * predator.L_predation +
                holling_type_II(tF, total_prey_time) * predator.F_predation +
                holling_type_II(tB, total_prey_time) * predator.B_predation +
                holling_type_II(tC, total_prey_time) * predator.C_predation
        )
        pred_fitness.append((predator, fitness))

    pL /= preds
    pF /= preds
    pB /= preds
    pC /= preds

    avg_predation = Env(X_L=pL, X_F=pF, X_B=pB, X_C=pC)
    prey_fitness = compute_prey_fitness(strategies=strategies, environment=avg_predation)
    return prey_fitness, pred_fitness

def evaluate_strategy(
    strategies: list[Strategy_prey],
    initial_state: Population_state,
    environment: Env,
    steps: int = 500, #initial 500
    burn_in: int = 1000, #initial 100
    r_F_L: int = 8,
    relocation_interval: int = 200,
    use_coevolution: bool = False,
    predators: list[Predator] = None,
) -> tuple[dict[str, float], list[tuple[Strategy_prey, float]], list[tuple[Predator, float]]]:

    """
    Runs simulation and returns:
    1. positional frequencies f
    2. prey fitness score
    """

    frequencies = run_simulation(
        initial_state=initial_state,
        strategies=strategies,
        steps=steps,
        r_F_L=r_F_L,
        relocation_interval=relocation_interval,
        use_coevolution = use_coevolution
    )

    fitness = []
    if not use_coevolution:
        fitness = compute_prey_fitness(
            strategies=strategies,
            environment=environment
        )

    pred_fitness = []
    if use_coevolution:
       fitness, pred_fitness = compute_pred_prey_fitness(
           strategies=strategies,
           predators=predators,
       )


    return frequencies, fitness, pred_fitness