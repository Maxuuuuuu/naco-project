# Imports 
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import random
import csv 
from pathlib import Path 

# Phase 1: Define the model components (prey strategy, environment, prey agent)
class Position(Enum):
    LEADER = "L"
    FOLLOWER = "F"
    BORDER = "B"
    CENTRE = "C"


@dataclass
class StrategyPrey:
    """
    Yang-style prey Strategy. 
    
    p_F_given_L: probability that a leader becomes a follower
    p_L_given_F: probability that a follower becomes a leader
    p_L_given_B: probability that a border individual becomes a leader
    
    """

    p_F_given_L: float
    p_L_given_F: float
    p_L_given_B: float

    def clip(self):
        """
          Keeps all strategy probabilities inside [0, 1].
        Important after mutation.
        """
        self.p_F_given_L = min(1.0, max(0.0, self.p_F_given_L))
        self.p_L_given_F = min(1.0, max(0.0, self.p_L_given_F))
        self.p_L_given_B = min(1.0, max(0.0, self.p_L_given_B))
        return self
    
    @staticmethod
    def random() -> "StrategyPrey":
        """
        Generates a random strategy.
        """
        return StrategyPrey(
            p_F_given_L=random.uniform(0, 1),
            p_L_given_F=random.uniform(0, 1),
            p_L_given_B=random.uniform(0, 1)
        )
    
@dataclass
class Environment:
    """
    Fixed Yang-style positional predation risk.

    Lower risk is better for prey.
    The environment should always be normalized before use.
    """

    X_L: float
    X_F: float
    X_B: float
    X_C: float

    def normalize(self) -> "Environment":
        total = self.X_L + self.X_F + self.X_B + self.X_C

        if total <= 0:
            raise ValueError("Total risk must be greater than 0 for normalization.")

        return Environment(
            X_L=self.X_L / total,
            X_F=self.X_F / total,
            X_B=self.X_B / total,
            X_C=self.X_C / total,
        )

    def as_dict(self) -> dict[Position, float]:
        env = self.normalize()

        return {
            Position.LEADER: env.X_L,
            Position.FOLLOWER: env.X_F,
            Position.BORDER: env.X_B,
            Position.CENTRE: env.X_C,
        }

# Phase 2: Define the prey agent and its position update rules
@dataclass
class PreyAgent:
    """
    One prey individual in Yang's abstract positional model.

    Each prey has:
    - a current position: L, F, B, or C
    - a strategy containing the three transition probabilities
    """

    strategy: StrategyPrey
    position: Position

    def update_position(self) -> None:
        """
        Implements Yang Table 1 position-change rules.

        Leader:
            attractive action -> Follower
            repulsive action  -> Leader

        Follower:
            attractive action -> Follower
            repulsive action  -> Leader

        Border:
            attractive action -> Border
            repulsive action  -> Leader

        Centre:
            always remains Centre
        """

        r = random.random()

        if self.position == Position.LEADER:
            if r < self.strategy.p_F_given_L:
                self.position = Position.FOLLOWER
            else:
                self.position = Position.LEADER
            return

        if self.position == Position.FOLLOWER:
            if r < self.strategy.p_L_given_F:
                self.position = Position.LEADER
            else:
                self.position = Position.FOLLOWER
            return

        if self.position == Position.BORDER:
            if r < self.strategy.p_L_given_B:
                self.position = Position.LEADER
            else:
                self.position = Position.BORDER
            return

        if self.position == Position.CENTRE:
            self.position = Position.CENTRE
            return

        raise ValueError(f"Unknown prey position: {self.position}")
    

# Phase 3 - Helper functions 
def count_positions(population: list[PreyAgent]) -> dict[Position, int]:
    """
    Counts how many prey are currently in each position.
    """

    counts = {
        Position.LEADER: 0,
        Position.FOLLOWER: 0,
        Position.BORDER: 0,
        Position.CENTRE: 0,
    }

    for prey in population:
        counts[prey.position] += 1

    return counts


def position_frequencies(population: list[PreyAgent]) -> dict[Position, float]:
    """
    Converts position counts into frequencies.
    """

    if len(population) == 0:
        raise ValueError("Cannot compute frequencies for an empty population.")

    counts = count_positions(population)
    total = len(population)

    return {
        position: count / total
        for position, count in counts.items()
    }


def enforce_mobile_group_constraint(
    population: list[PreyAgent],
    r_F_per_L: int = 8,
) -> None:
    """
    Enforces Yang's mobile-group constraint.

    A mobile group cannot have too many followers relative to leaders.

    If:
        number of followers > r_F_per_L * number of leaders

    then excess followers are converted to BORDER individuals.
    """

    counts = count_positions(population)

    n_leaders = counts[Position.LEADER]
    n_followers = counts[Position.FOLLOWER]

    max_followers = r_F_per_L * n_leaders

    if n_followers <= max_followers:
        return

    excess_followers = n_followers - max_followers

    follower_agents = [
        prey for prey in population
        if prey.position == Position.FOLLOWER
    ]

    random.shuffle(follower_agents)

    for prey in follower_agents[:excess_followers]:
        prey.position = Position.BORDER


def calculate_target_centres(herd_size: int) -> int:
    """
    Calculates the number of centre individuals allowed in a cohesive herd.

    Yang approximates the cohesive herd as cube-shaped.

    If the herd has fewer than 8 individuals, no centre can exist.
    """

    if herd_size < 8:
        return 0

    target_centres = int((herd_size ** (1 / 3) - 2) ** 3)

    return max(0, min(target_centres, herd_size))


def enforce_herd_geometry_constraint(population: list[PreyAgent]) -> None:
    """
    Enforces the BORDER/CENTRE geometric constraint.

    Only prey currently in BORDER or CENTRE are treated as part of the cohesive herd.
    The allowed number of CENTRE individuals depends on the total herd size.
    The remaining herd individuals become BORDER.
    """

    herd_agents = [
        prey for prey in population
        if prey.position in {Position.BORDER, Position.CENTRE}
    ]

    herd_size = len(herd_agents)
    target_centres = calculate_target_centres(herd_size)

    random.shuffle(herd_agents)

    for i, prey in enumerate(herd_agents):
        if i < target_centres:
            prey.position = Position.CENTRE
        else:
            prey.position = Position.BORDER


def enforce_geometric_constraints(
    population: list[PreyAgent],
    r_F_per_L: int = 8,
) -> None:
    """
    Applies both geometric constraints.

    Order matters:
    1. First enforce mobile-group constraint.
       This may move excess followers into the border.
    2. Then enforce herd geometry.
       This adjusts BORDER and CENTRE counts.
    """

    enforce_mobile_group_constraint(
        population=population,
        r_F_per_L=r_F_per_L,
    )

    enforce_herd_geometry_constraint(population)


# Phase 4 - A simulation episode 
def random_position() -> Position:
    """
    Randomly assigns one of Yang's four abstract positions.
    """

    return random.choice(
        [
            Position.LEADER,
            Position.FOLLOWER,
            Position.BORDER,
            Position.CENTRE,
        ]
    )


def create_random_population(population_size: int) -> list[PreyAgent]:
    """
    Creates a population of prey agents.

    Each prey receives:
    - a random Yang-style strategy
    - a random starting position
    """

    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")

    return [
        PreyAgent(
            strategy=StrategyPrey.random(),
            position=random_position(),
        )
        for _ in range(population_size)
    ]


def empty_position_counter() -> dict[Position, int]:
    """
    Creates an empty counter for the four Yang positions.
    """

    return {
        Position.LEADER: 0,
        Position.FOLLOWER: 0,
        Position.BORDER: 0,
        Position.CENTRE: 0,
    }

# This function was updated before Phase 7
def run_position_episode(
    population: list[PreyAgent],
    steps: int,
    r_F_per_L: int = 8,
    relocation_interval: int | None = None,
) -> list[dict[Position, float]]:
    """
    Runs one positional simulation episode.

    For each step:
    1. Optionally relocate/randomize positions at fixed intervals.
    2. Every prey updates its position using Yang Table 1.
    3. Geometric constraints are enforced.
    4. Each prey's current position is recorded.

    relocation_interval:
        If None, no relocation is used.
        If an integer, positions are randomized every relocation_interval steps.

    Returns:
        A list of individual positional-frequency dictionaries.
    """

    if len(population) == 0:
        raise ValueError("Cannot run an episode with an empty population.")

    if steps <= 0:
        raise ValueError("Episode steps must be greater than 0.")

    if relocation_interval is not None and relocation_interval <= 0:
        raise ValueError("Relocation interval must be greater than 0 or None.")

    position_histories = [
        empty_position_counter()
        for _ in population
    ]

    for step in range(steps):
        if (
            relocation_interval is not None
            and step > 0
            and step % relocation_interval == 0
        ):
            randomize_population_positions(population)

        for prey in population:
            prey.update_position()

        enforce_geometric_constraints(
            population=population,
            r_F_per_L=r_F_per_L,
        )

        for prey_index, prey in enumerate(population):
            position_histories[prey_index][prey.position] += 1

    individual_frequencies = []

    for history in position_histories:
        frequencies = {
            position: count / steps
            for position, count in history.items()
        }

        individual_frequencies.append(frequencies)

    return individual_frequencies

def average_individual_frequencies(
    individual_frequencies: list[dict[Position, float]]
) -> dict[Position, float]:
    """
    Computes the population-average positional frequencies.

    This gives the overall f = (f_L, f_F, f_B, f_C)
    for the episode.
    """

    if len(individual_frequencies) == 0:
        raise ValueError("Cannot average an empty list of frequencies.")

    totals = {
        Position.LEADER: 0.0,
        Position.FOLLOWER: 0.0,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    for frequencies in individual_frequencies:
        for position in totals:
            totals[position] += frequencies[position]

    number_of_individuals = len(individual_frequencies)

    return {
        position: total / number_of_individuals
        for position, total in totals.items()
    }

"""
population = create_random_population(1000)

individual_frequencies = run_position_episode(
    population=population,
    steps=100,
    r_F_per_L=8,
)

average_frequencies = average_individual_frequencies(
    individual_frequencies
)

print(average_frequencies) 

IGNORE THE CODE ABOVE. 
"""

# Phase 5 - Risk and fitness calculations 
# Something to keep in mind here is - Fitness = -risk 

def compute_individual_risk(
    frequencies: dict[Position, float],
    environment: Environment,
) -> float:
    """
    Computes Yang-style individual positional risk.

    risk = X^T f

    Lower risk is better for prey.
    """

    env = environment.as_dict()

    return (
        env[Position.LEADER] * frequencies[Position.LEADER]
        + env[Position.FOLLOWER] * frequencies[Position.FOLLOWER]
        + env[Position.BORDER] * frequencies[Position.BORDER]
        + env[Position.CENTRE] * frequencies[Position.CENTRE]
    )


def compute_individual_fitness(
    frequencies: dict[Position, float],
    environment: Environment,
) -> float:
    """
    Converts risk into fitness.

    Since lower risk is better, fitness is negative risk.
    Higher fitness is better.
    """

    risk = compute_individual_risk(
        frequencies=frequencies,
        environment=environment,
    )

    return -risk


def compute_population_risks(
    individual_frequencies: list[dict[Position, float]],
    environment: Environment,
) -> list[float]:
    """
    Computes risk for every prey individual.
    """

    return [
        compute_individual_risk(
            frequencies=frequencies,
            environment=environment,
        )
        for frequencies in individual_frequencies
    ]


def compute_population_fitnesses(
    individual_frequencies: list[dict[Position, float]],
    environment: Environment,
) -> list[float]:
    """
    Computes fitness for every prey individual.
    """

    return [
        compute_individual_fitness(
            frequencies=frequencies,
            environment=environment,
        )
        for frequencies in individual_frequencies
    ]


def compute_mean_risk(risks: list[float]) -> float:
    """
    Computes mean population risk.
    """

    if len(risks) == 0:
        raise ValueError("Cannot compute mean risk for an empty risk list.")

    return sum(risks) / len(risks)

# Real run - INGORE THIS PART, JUST A TEST
# if __name__ == "__main__": 
#     environment = Environment(
#         X_L=0.18,
#         X_F=0.04,
#         X_B=0.53,
#         X_C=0.25,
#     )

#     population = create_random_population(1000)

#     individual_frequencies = run_position_episode(
#         population=population,
#         steps=100,
#         r_F_per_L=8,
#     )

#     average_frequencies = average_individual_frequencies(
#         individual_frequencies
#     )

#     risks = compute_population_risks(
#         individual_frequencies=individual_frequencies,
#         environment=environment,
#     )

#     mean_risk = compute_mean_risk(risks)

#     print("Average frequencies:", average_frequencies)
#     print("Mean risk:", mean_risk)


# Phase 6 - GA search update
def clone_strategy(strategy: StrategyPrey) -> StrategyPrey:
    """
    Creates a copy of a prey strategy.
    """

    return StrategyPrey(
        p_F_given_L=strategy.p_F_given_L,
        p_L_given_F=strategy.p_L_given_F,
        p_L_given_B=strategy.p_L_given_B,
    )


def mutate_strategy(
    strategy: StrategyPrey,
    mutation_std: float = 0.02,
) -> StrategyPrey:
    """
    Creates a mutated offspring strategy.

    In Yang's GA search:
        c_offspring = c_parent + epsilon
        epsilon ~ N(0, 0.02)

    Here, independent Gaussian noise is applied to each of the
    three strategy probabilities.
    """

    offspring_strategy = StrategyPrey(
        p_F_given_L=strategy.p_F_given_L + random.gauss(0.0, mutation_std),
        p_L_given_F=strategy.p_L_given_F + random.gauss(0.0, mutation_std),
        p_L_given_B=strategy.p_L_given_B + random.gauss(0.0, mutation_std),
    )

    return offspring_strategy.clip()


def calculate_replacement_count(
    population_size: int,
    replacement_rate: float = 0.10,
) -> int:
    """
    Calculates how many prey are replaced per GA search generation.

    Yang uses 10% replacement per generation.
    """

    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")

    if not 0.0 < replacement_rate <= 1.0:
        raise ValueError("Replacement rate must be in the interval (0, 1].")

    return max(1, int(round(population_size * replacement_rate)))


def tournament_select_parent(
    population: list[PreyAgent],
    fitnesses: list[float],
    tournament_size: int = 4,
) -> PreyAgent:
    """
    Selects one parent using tournament selection.

    Higher fitness is better.

    Since fitness = -risk, this means prey with lower risk are preferred.
    """

    if len(population) == 0:
        raise ValueError("Cannot select from an empty population.")

    if len(population) != len(fitnesses):
        raise ValueError("Population and fitness list must have the same length.")

    if tournament_size <= 0:
        raise ValueError("Tournament size must be greater than 0.")

    candidate_indices = random.sample(
        range(len(population)),
        k=min(tournament_size, len(population)),
    )

    best_index = max(
        candidate_indices,
        key=lambda index: fitnesses[index],
    )

    return population[best_index]


def get_replacement_indices(
    fitnesses: list[float],
    replacement_count: int,
) -> list[int]:
    """
    Returns the indices of prey that should be replaced.

    Lowest fitness means highest risk, so those individuals are replaced first.
    """

    if len(fitnesses) == 0:
        raise ValueError("Cannot replace from an empty fitness list.")

    if replacement_count <= 0:
        raise ValueError("Replacement count must be greater than 0.")

    if replacement_count > len(fitnesses):
        raise ValueError("Replacement count cannot exceed population size.")

    sorted_indices = sorted(
        range(len(fitnesses)),
        key=lambda index: fitnesses[index],
    )

    return sorted_indices[:replacement_count]


def ga_search_update(
    population: list[PreyAgent],
    individual_frequencies: list[dict[Position, float]],
    environment: Environment,
    replacement_rate: float = 0.10,
    tournament_size: int = 4,
    mutation_std: float = 0.02,
) -> None:
    """
    Performs one Yang-style GA search update.

    This should be understood as a numerical search step for a candidate
    resident strategy c*, not as the final ex-post ESS validation.

    Steps:
    1. Compute fitness for every prey.
    2. Select the lowest-fitness prey for replacement.
    3. Select parents using tournament selection.
    4. Create offspring by mutating parent strategies.
    5. Replace low-fitness prey with offspring.

    The ex-post ESS/invasion check will be added separately after this phase.
    """

    if len(population) == 0:
        raise ValueError("Cannot update an empty population.")

    if len(population) != len(individual_frequencies):
        raise ValueError(
            "Population and individual_frequencies must have the same length."
        )

    fitnesses = compute_population_fitnesses(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    replacement_count = calculate_replacement_count(
        population_size=len(population),
        replacement_rate=replacement_rate,
    )

    replacement_indices = get_replacement_indices(
        fitnesses=fitnesses,
        replacement_count=replacement_count,
    )

    # Important:
    # Parent selection should be based on the old generation, not on partially
    # replaced individuals inside the same update step.
    original_population = list(population)

    for replacement_index in replacement_indices:
        parent = tournament_select_parent(
            population=original_population,
            fitnesses=fitnesses,
            tournament_size=tournament_size,
        )

        offspring_strategy = mutate_strategy(
            strategy=parent.strategy,
            mutation_std=mutation_std,
        )

        population[replacement_index] = PreyAgent(
            strategy=offspring_strategy,
            position=random_position(),
        )


def mean_strategy(population: list[PreyAgent]) -> StrategyPrey:
    """
    Computes the population-average strategy.
    """

    if len(population) == 0:
        raise ValueError("Cannot compute mean strategy for an empty population.")

    return StrategyPrey(
        p_F_given_L=sum(
            prey.strategy.p_F_given_L for prey in population
        ) / len(population),
        p_L_given_F=sum(
            prey.strategy.p_L_given_F for prey in population
        ) / len(population),
        p_L_given_B=sum(
            prey.strategy.p_L_given_B for prey in population
        ) / len(population),
    )

# Phase 6.5 - Ex-post ESS 
@dataclass
class ESSComparison:
    """
    Stores the result of one resident-versus-mutant invasion check.

    resident_strategy:
        Candidate resident strategy c*

    mutant_strategy:
        Alternative invading strategy c'

    resident_mean_risk / mutant_mean_risk:
        Mean X^T f risk for each group

    resident_mean_fitness / mutant_mean_fitness:
        Mean -risk fitness for each group

    is_resident_stable:
        True if resident fitness is higher than mutant fitness.
    """

    resident_strategy: StrategyPrey
    mutant_strategy: StrategyPrey
    resident_count: int
    mutant_count: int
    resident_mean_risk: float
    mutant_mean_risk: float
    resident_mean_fitness: float
    mutant_mean_fitness: float
    resident_average_frequencies: dict[Position, float]
    mutant_average_frequencies: dict[Position, float]
    overall_average_frequencies: dict[Position, float]
    is_resident_stable: bool


def create_population_from_strategy(
    strategy: StrategyPrey,
    population_size: int,
) -> list[PreyAgent]:
    """
    Creates a population where every prey has the same strategy.

    This is useful for constructing resident and mutant subpopulations.
    """

    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")

    return [
        PreyAgent(
            strategy=clone_strategy(strategy),
            position=random_position(),
        )
        for _ in range(population_size)
    ]


def create_perturbed_population(
    resident_strategy: StrategyPrey,
    mutant_strategy: StrategyPrey,
    population_size: int,
    mutant_fraction: float = 0.05,
) -> tuple[list[PreyAgent], list[str]]:
    """
    Creates a Yang-style perturbed population.

    Most individuals use the resident strategy c*.
    A small fraction use the mutant strategy c'.

    Returns:
        population:
            List of PreyAgent objects.

        labels:
            Same length as population.
            Each label is either "resident" or "mutant".
    """

    if population_size <= 1:
        raise ValueError("Population size must be greater than 1.")

    if not 0.0 < mutant_fraction < 1.0:
        raise ValueError("Mutant fraction must be in the interval (0, 1).")

    mutant_count = max(1, int(round(population_size * mutant_fraction)))
    mutant_count = min(mutant_count, population_size - 1)

    resident_count = population_size - mutant_count

    resident_population = create_population_from_strategy(
        strategy=resident_strategy,
        population_size=resident_count,
    )

    mutant_population = create_population_from_strategy(
        strategy=mutant_strategy,
        population_size=mutant_count,
    )

    combined = (
        [(prey, "resident") for prey in resident_population]
        + [(prey, "mutant") for prey in mutant_population]
    )

    random.shuffle(combined)

    population = [prey for prey, _ in combined]
    labels = [label for _, label in combined]

    return population, labels


def mean(values: list[float]) -> float:
    """
    Computes the mean of a non-empty list.
    """

    if len(values) == 0:
        raise ValueError("Cannot compute mean of an empty list.")

    return sum(values) / len(values)


def values_for_label(
    values: list[float],
    labels: list[str],
    target_label: str,
) -> list[float]:
    """
    Selects values belonging to a specific label.
    """

    if len(values) != len(labels):
        raise ValueError("Values and labels must have the same length.")

    return [
        value
        for value, label in zip(values, labels)
        if label == target_label
    ]


def frequencies_for_label(
    individual_frequencies: list[dict[Position, float]],
    labels: list[str],
    target_label: str,
) -> list[dict[Position, float]]:
    """
    Selects individual frequency dictionaries belonging to a specific label.
    """

    if len(individual_frequencies) != len(labels):
        raise ValueError(
            "Individual frequencies and labels must have the same length."
        )

    return [
        frequencies
        for frequencies, label in zip(individual_frequencies, labels)
        if label == target_label
    ]


def evaluate_resident_against_mutant(
    resident_strategy: StrategyPrey,
    mutant_strategy: StrategyPrey,
    environment: Environment,
    population_size: int = 1000,
    mutant_fraction: float = 0.05,
    steps: int = 1000,
    r_F_per_L: int = 8,
    fitness_tolerance: float = 0.0,
) -> ESSComparison:
    """
    Performs one ex-post resident-versus-mutant invasion check.

    This is the key Phase 6.5 step.

    Procedure:
    1. Create a perturbed population:
        mostly resident strategy c*
        small mutant fraction c'

    2. Run only the positional episode.
       No GA replacement happens here.

    3. Compute resident and mutant risks separately.

    4. Check whether resident has higher fitness than mutant.

    Since fitness = -risk:
        resident stable means:
            resident_mean_fitness > mutant_mean_fitness

        equivalently:
            resident_mean_risk < mutant_mean_risk
    """

    if fitness_tolerance < 0:
        raise ValueError("Fitness tolerance cannot be negative.")

    population, labels = create_perturbed_population(
        resident_strategy=resident_strategy,
        mutant_strategy=mutant_strategy,
        population_size=population_size,
        mutant_fraction=mutant_fraction,
    )

    individual_frequencies = run_position_episode(
        population=population,
        steps=steps,
        r_F_per_L=r_F_per_L,
    )

    risks = compute_population_risks(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    fitnesses = [-risk for risk in risks]

    resident_risks = values_for_label(
        values=risks,
        labels=labels,
        target_label="resident",
    )

    mutant_risks = values_for_label(
        values=risks,
        labels=labels,
        target_label="mutant",
    )

    resident_fitnesses = values_for_label(
        values=fitnesses,
        labels=labels,
        target_label="resident",
    )

    mutant_fitnesses = values_for_label(
        values=fitnesses,
        labels=labels,
        target_label="mutant",
    )

    resident_frequency_list = frequencies_for_label(
        individual_frequencies=individual_frequencies,
        labels=labels,
        target_label="resident",
    )

    mutant_frequency_list = frequencies_for_label(
        individual_frequencies=individual_frequencies,
        labels=labels,
        target_label="mutant",
    )

    resident_mean_fitness = mean(resident_fitnesses)
    mutant_mean_fitness = mean(mutant_fitnesses)

    is_resident_stable = (
        resident_mean_fitness > mutant_mean_fitness + fitness_tolerance
    )

    return ESSComparison(
        resident_strategy=clone_strategy(resident_strategy),
        mutant_strategy=clone_strategy(mutant_strategy),
        resident_count=len(resident_risks),
        mutant_count=len(mutant_risks),
        resident_mean_risk=mean(resident_risks),
        mutant_mean_risk=mean(mutant_risks),
        resident_mean_fitness=resident_mean_fitness,
        mutant_mean_fitness=mutant_mean_fitness,
        resident_average_frequencies=average_individual_frequencies(
            resident_frequency_list
        ),
        mutant_average_frequencies=average_individual_frequencies(
            mutant_frequency_list
        ),
        overall_average_frequencies=average_individual_frequencies(
            individual_frequencies
        ),
        is_resident_stable=is_resident_stable,
    )


def evaluate_resident_against_many_mutants(
    resident_strategy: StrategyPrey,
    mutant_strategies: list[StrategyPrey],
    environment: Environment,
    population_size: int = 1000,
    mutant_fraction: float = 0.05,
    steps: int = 1000,
    r_F_per_L: int = 8,
    fitness_tolerance: float = 0.0,
) -> list[ESSComparison]:
    """
    Tests one candidate resident strategy against many mutant strategies.

    A resident is a stronger ESS candidate if it is stable against all tested mutants.
    """

    if len(mutant_strategies) == 0:
        raise ValueError("At least one mutant strategy is required.")

    return [
        evaluate_resident_against_mutant(
            resident_strategy=resident_strategy,
            mutant_strategy=mutant_strategy,
            environment=environment,
            population_size=population_size,
            mutant_fraction=mutant_fraction,
            steps=steps,
            r_F_per_L=r_F_per_L,
            fitness_tolerance=fitness_tolerance,
        )
        for mutant_strategy in mutant_strategies
    ]


def is_stable_against_all_mutants(
    comparisons: list[ESSComparison],
) -> bool:
    """
    Returns True only if the resident strategy beats every tested mutant.
    """

    if len(comparisons) == 0:
        raise ValueError("Comparison list cannot be empty.")

    return all(
        comparison.is_resident_stable
        for comparison in comparisons
    )

# if __name__ == "__main__":
#     environment = Environment(
#         X_L=0.18,
#         X_F=0.04,
#         X_B=0.53,
#         X_C=0.25,
#     )

#     resident_strategy = StrategyPrey(
#         p_F_given_L=1.0,
#         p_L_given_F=0.01,
#         p_L_given_B=1.0,
#     )

#     mutant_strategy = StrategyPrey(
#         p_F_given_L=0.0,
#         p_L_given_F=1.0,
#         p_L_given_B=0.0,
#     )

#     comparison = evaluate_resident_against_mutant(
#         resident_strategy=resident_strategy,
#         mutant_strategy=mutant_strategy,
#         environment=environment,
#         population_size=1000,
#         mutant_fraction=0.05,
#         steps=1000,
#         r_F_per_L=8,
#     )

#     print("Resident mean risk:", comparison.resident_mean_risk)
#     print("Mutant mean risk:", comparison.mutant_mean_risk)
#     print("Resident mean fitness:", comparison.resident_mean_fitness)
#     print("Mutant mean fitness:", comparison.mutant_mean_fitness)
#     print("Resident stable:", comparison.is_resident_stable)
#     print("Overall frequencies:", comparison.overall_average_frequencies)

# Helper function for Phase 7 
def randomize_population_positions(population: list[PreyAgent]) -> None:
    """
    Randomly reassigns positions for all prey.

    This is used to reduce bias from initial position allocation during
    longer positional episodes.
    """

    if len(population) == 0:
        raise ValueError("Cannot randomize positions for an empty population.")

    for prey in population:
        prey.position = random_position()

# Phase 7 GA search runner 
@dataclass
class GenerationRecord:
    """
    Stores summary results for one GA search generation.
    """

    generation: int
    mean_risk: float
    mean_fitness: float
    mean_strategy: StrategyPrey
    average_frequencies: dict[Position, float]
    collective_mobility: float


@dataclass
class GASearchResult:
    """
    Stores the output of a Yang-style GA search run.

    final_strategy:
        Candidate resident strategy c* found by the GA search.

    final_average_frequencies:
        Final population-average positional frequencies.

    records:
        Generation-by-generation summaries.
    """

    environment: Environment
    population_size: int
    generations: int
    episode_steps: int
    replacement_rate: float
    tournament_size: int
    mutation_std: float
    r_F_per_L: int
    relocation_interval: int | None
    final_strategy: StrategyPrey
    final_average_frequencies: dict[Position, float]
    final_mean_risk: float
    final_mean_fitness: float
    records: list[GenerationRecord]
    final_population: list[PreyAgent]


def compute_collective_mobility(
    frequencies: dict[Position, float],
) -> float:
    """
    Collective mobility is the proportion of individuals/time in mobile-group
    positions.

    CM = f_L + f_F
    """

    return frequencies[Position.LEADER] + frequencies[Position.FOLLOWER]


def create_generation_record(
    generation: int,
    population: list[PreyAgent],
    individual_frequencies: list[dict[Position, float]],
    environment: Environment,
) -> GenerationRecord:
    """
    Creates one generation-level summary record.
    """

    risks = compute_population_risks(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    fitnesses = compute_population_fitnesses(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    average_frequencies = average_individual_frequencies(
        individual_frequencies
    )

    return GenerationRecord(
        generation=generation,
        mean_risk=mean(risks),
        mean_fitness=mean(fitnesses),
        mean_strategy=mean_strategy(population),
        average_frequencies=average_frequencies,
        collective_mobility=compute_collective_mobility(
            average_frequencies
        ),
    )


def run_ga_search(
    environment: Environment,
    population_size: int = 1000,
    generations: int = 3000,
    episode_steps: int = 1000,
    replacement_rate: float = 0.10,
    tournament_size: int = 4,
    mutation_std: float = 0.02,
    r_F_per_L: int = 8,
    relocation_interval: int | None = 200,
    record_interval: int = 1,
    seed: int | None = None,
) -> GASearchResult:
    """
    Runs Yang-style GA search for a candidate resident strategy c*.

    Important:
    This is a numerical search procedure for a candidate resident strategy.
    It should not be described as the full biological evolutionary trajectory.
    The ex-post ESS/invasion check is handled separately by Phase 6.5.
    """

    if population_size <= 0:
        raise ValueError("Population size must be greater than 0.")

    if generations <= 0:
        raise ValueError("Generations must be greater than 0.")

    if episode_steps <= 0:
        raise ValueError("Episode steps must be greater than 0.")

    if record_interval <= 0:
        raise ValueError("Record interval must be greater than 0.")

    if seed is not None:
        random.seed(seed)

    environment = environment.normalize()

    population = create_random_population(
        population_size=population_size
    )

    records: list[GenerationRecord] = []

    last_individual_frequencies: list[dict[Position, float]] | None = None

    for generation in range(generations):
        individual_frequencies = run_position_episode(
            population=population,
            steps=episode_steps,
            r_F_per_L=r_F_per_L,
            relocation_interval=relocation_interval,
        )

        last_individual_frequencies = individual_frequencies

        if generation % record_interval == 0 or generation == generations - 1:
            record = create_generation_record(
                generation=generation,
                population=population,
                individual_frequencies=individual_frequencies,
                environment=environment,
            )

            records.append(record)

        ga_search_update(
            population=population,
            individual_frequencies=individual_frequencies,
            environment=environment,
            replacement_rate=replacement_rate,
            tournament_size=tournament_size,
            mutation_std=mutation_std,
        )

    # Evaluate final population after the last update.
    final_individual_frequencies = run_position_episode(
        population=population,
        steps=episode_steps,
        r_F_per_L=r_F_per_L,
        relocation_interval=relocation_interval,
    )

    final_risks = compute_population_risks(
        individual_frequencies=final_individual_frequencies,
        environment=environment,
    )

    final_fitnesses = compute_population_fitnesses(
        individual_frequencies=final_individual_frequencies,
        environment=environment,
    )

    final_average_frequencies = average_individual_frequencies(
        final_individual_frequencies
    )

    return GASearchResult(
        environment=environment,
        population_size=population_size,
        generations=generations,
        episode_steps=episode_steps,
        replacement_rate=replacement_rate,
        tournament_size=tournament_size,
        mutation_std=mutation_std,
        r_F_per_L=r_F_per_L,
        relocation_interval=relocation_interval,
        final_strategy=mean_strategy(population),
        final_average_frequencies=final_average_frequencies,
        final_mean_risk=mean(final_risks),
        final_mean_fitness=mean(final_fitnesses),
        records=records,
        final_population=population,
    )

# Phase 8 - Figure 1 and Figure 2 reproduction layer
@dataclass
class FigureReproductionTarget:
    """
    Stores one Yang figure-reproduction target.

    expected_strategy is c = (p(F|L), p(L|F), p(L|B)).
    expected_frequencies are f = (f_L, f_F, f_B, f_C).
    expected_cm is optional because Figure 1 reports frequencies directly,
    while Figure 2 also discusses collective mobility explicitly.
    """

    name: str
    environment: Environment
    expected_strategy: StrategyPrey
    expected_frequencies: dict[Position, float]
    expected_cm: float | None = None
    use_late_generation_average: bool = False
    late_generation_fraction: float = 1 / 3


@dataclass
class LateGenerationAverage:
    """
    Stores averaged values from the last part of a GA search.
    """

    start_generation: int
    end_generation: int
    number_of_records: int
    mean_strategy: StrategyPrey
    mean_frequencies: dict[Position, float]
    mean_collective_mobility: float
    mean_risk: float
    mean_fitness: float


@dataclass
class FigureReproductionResult:
    """
    Stores the result of reproducing one Yang figure target.
    """

    target: FigureReproductionTarget
    ga_result: GASearchResult
    observed_strategy: StrategyPrey
    observed_frequencies: dict[Position, float]
    observed_cm: float
    strategy_absolute_errors: dict[str, float]
    frequency_absolute_errors: dict[Position, float]
    mean_strategy_absolute_error: float
    mean_frequency_absolute_error: float
    cm_absolute_error: float | None
    late_generation_average: LateGenerationAverage | None = None


FIGURE_1A_TARGET = FigureReproductionTarget(
    name="Figure 1A",
    environment=Environment(0.18, 0.04, 0.53, 0.25),
    expected_strategy=StrategyPrey(1.0, 0.01, 1.0),
    expected_frequencies={
        Position.LEADER: 0.16,
        Position.FOLLOWER: 0.61,
        Position.BORDER: 0.15,
        Position.CENTRE: 0.08,
    },
    expected_cm=0.77,
    use_late_generation_average=False,
)


FIGURE_1B_TARGET = FigureReproductionTarget(
    name="Figure 1B",
    environment=Environment(0.45, 0.22, 0.24, 0.09),
    expected_strategy=StrategyPrey(1.0, 0.0, 0.0),
    expected_frequencies={
        Position.LEADER: 0.00,
        Position.FOLLOWER: 0.00,
        Position.BORDER: 0.49,
        Position.CENTRE: 0.51,
    },
    expected_cm=0.00,
    use_late_generation_average=False,
)


FIGURE_2A_TARGET = FigureReproductionTarget(
    name="Figure 2A",
    environment=Environment(0.38, 0.15, 0.11, 0.36),
    expected_strategy=StrategyPrey(1.0, 0.0, 0.27),
    expected_frequencies={
        Position.LEADER: 0.07,
        Position.FOLLOWER: 0.50,
        Position.BORDER: 0.26,
        Position.CENTRE: 0.17,
    },
    expected_cm=0.57,
    use_late_generation_average=True,
)


FIGURE_2B_TARGET = FigureReproductionTarget(
    name="Figure 2B",
    environment=Environment(0.25, 0.19, 0.03, 0.53),
    expected_strategy=StrategyPrey(0.94, 0.01, 0.10),
    expected_frequencies={
        Position.LEADER: 0.04,
        Position.FOLLOWER: 0.28,
        Position.BORDER: 0.37,
        Position.CENTRE: 0.31,
    },
    expected_cm=0.31,
    use_late_generation_average=True,
)


FIGURE_REPRODUCTION_TARGETS = [
    FIGURE_1A_TARGET,
    FIGURE_1B_TARGET,
    FIGURE_2A_TARGET,
    FIGURE_2B_TARGET,
]

FIGURE_REPRODUCTION_SMALL_SETTINGS = {
    "population_size": 30,
    "generations": 3,
    "episode_steps": 20,
    "replacement_rate": 0.10,
    "tournament_size": 4,
    "mutation_std": 0.02,
    "r_F_per_L": 8,
    "relocation_interval": 10,
    "record_interval": 1,
}


FIGURE_REPRODUCTION_MEDIUM_SETTINGS = {
    "population_size": 500,
    "generations": 500,
    "episode_steps": 1000,
    "replacement_rate": 0.10,
    "tournament_size": 4,
    "mutation_std": 0.02,
    "r_F_per_L": 8,
    "relocation_interval": 200,
    "record_interval": 10,
}

def strategy_to_tuple(strategy: StrategyPrey) -> tuple[float, float, float]:
    """
    Converts c = (p(F|L), p(L|F), p(L|B)) into a tuple.
    """

    return (
        strategy.p_F_given_L,
        strategy.p_L_given_F,
        strategy.p_L_given_B,
    )


def strategy_absolute_errors(
    observed: StrategyPrey,
    expected: StrategyPrey,
) -> dict[str, float]:
    """
    Computes component-wise absolute error for the strategy vector c.
    """

    return {
        "p_F_given_L": abs(observed.p_F_given_L - expected.p_F_given_L),
        "p_L_given_F": abs(observed.p_L_given_F - expected.p_L_given_F),
        "p_L_given_B": abs(observed.p_L_given_B - expected.p_L_given_B),
    }


def frequency_absolute_errors(
    observed: dict[Position, float],
    expected: dict[Position, float],
) -> dict[Position, float]:
    """
    Computes component-wise absolute error for positional frequencies.
    """

    return {
        position: abs(observed[position] - expected[position])
        for position in expected
    }


def mean_absolute_error(values: dict) -> float:
    """
    Computes the mean absolute error from a dictionary of errors.
    """

    if len(values) == 0:
        raise ValueError("Cannot compute mean absolute error for an empty dictionary.")

    return sum(values.values()) / len(values)

def average_strategies(strategies: list[StrategyPrey]) -> StrategyPrey:
    """
    Computes the average of several StrategyPrey objects.
    """

    if len(strategies) == 0:
        raise ValueError("Cannot average an empty strategy list.")

    return StrategyPrey(
        p_F_given_L=sum(s.p_F_given_L for s in strategies) / len(strategies),
        p_L_given_F=sum(s.p_L_given_F for s in strategies) / len(strategies),
        p_L_given_B=sum(s.p_L_given_B for s in strategies) / len(strategies),
    )


def average_frequency_dicts(
    frequency_dicts: list[dict[Position, float]],
) -> dict[Position, float]:
    """
    Computes the average of several positional-frequency dictionaries.
    """

    if len(frequency_dicts) == 0:
        raise ValueError("Cannot average an empty frequency list.")

    return {
        position: sum(frequencies[position] for frequencies in frequency_dicts) / len(frequency_dicts)
        for position in [
            Position.LEADER,
            Position.FOLLOWER,
            Position.BORDER,
            Position.CENTRE,
        ]
    }


def average_late_generation_records(
    records: list[GenerationRecord],
    late_generation_fraction: float = 1 / 3,
) -> LateGenerationAverage:
    """
    Averages the final fraction of recorded GA generations.

    This is mainly used for Figure 2 cases, where Yang reports weaker
    strategy convergence when X_C > X_B, but relatively stable population-level
    emergent patterns.
    """

    if len(records) == 0:
        raise ValueError("Cannot average an empty record list.")

    if not 0 < late_generation_fraction <= 1:
        raise ValueError("late_generation_fraction must be in the interval (0, 1].")

    number_of_late_records = max(1, int(len(records) * late_generation_fraction))
    late_records = records[-number_of_late_records:]

    return LateGenerationAverage(
        start_generation=late_records[0].generation,
        end_generation=late_records[-1].generation,
        number_of_records=len(late_records),
        mean_strategy=average_strategies(
            [record.mean_strategy for record in late_records]
        ),
        mean_frequencies=average_frequency_dicts(
            [record.average_frequencies for record in late_records]
        ),
        mean_collective_mobility=mean(
            [record.collective_mobility for record in late_records]
        ),
        mean_risk=mean(
            [record.mean_risk for record in late_records]
        ),
        mean_fitness=mean(
            [record.mean_fitness for record in late_records]
        ),
    )

def reproduce_figure_target(
    target: FigureReproductionTarget,
    seed: int | None = None,
    **ga_settings,
) -> FigureReproductionResult:
    """
    Runs run_ga_search() for one Yang figure target and compares the output
    with Yang's reported strategy/frequency values.
    """

    ga_result = run_ga_search(
        environment=target.environment,
        seed=seed,
        **ga_settings,
    )

    late_average = None

    if target.use_late_generation_average:
        late_average = average_late_generation_records(
            records=ga_result.records,
            late_generation_fraction=target.late_generation_fraction,
        )
        observed_strategy = late_average.mean_strategy
        observed_frequencies = late_average.mean_frequencies
        observed_cm = late_average.mean_collective_mobility
    else:
        observed_strategy = ga_result.final_strategy
        observed_frequencies = ga_result.final_average_frequencies
        observed_cm = compute_collective_mobility(observed_frequencies)

    s_errors = strategy_absolute_errors(
        observed=observed_strategy,
        expected=target.expected_strategy,
    )

    f_errors = frequency_absolute_errors(
        observed=observed_frequencies,
        expected=target.expected_frequencies,
    )

    cm_error = None
    if target.expected_cm is not None:
        cm_error = abs(observed_cm - target.expected_cm)

    return FigureReproductionResult(
        target=target,
        ga_result=ga_result,
        observed_strategy=observed_strategy,
        observed_frequencies=observed_frequencies,
        observed_cm=observed_cm,
        strategy_absolute_errors=s_errors,
        frequency_absolute_errors=f_errors,
        mean_strategy_absolute_error=mean_absolute_error(s_errors),
        mean_frequency_absolute_error=mean_absolute_error(f_errors),
        cm_absolute_error=cm_error,
        late_generation_average=late_average,
    )


def run_figure_reproduction_suite(
    targets: list[FigureReproductionTarget] | None = None,
    settings: dict | None = None,
    seed: int | None = 42,
) -> list[FigureReproductionResult]:
    """
    Runs the reproduction wrapper for all selected targets.
    """

    if targets is None:
        targets = FIGURE_REPRODUCTION_TARGETS

    if settings is None:
        settings = FIGURE_REPRODUCTION_MEDIUM_SETTINGS

    results = []

    for index, target in enumerate(targets):
        target_seed = None if seed is None else seed + index
        result = reproduce_figure_target(
            target=target,
            seed=target_seed,
            **settings,
        )
        results.append(result)

    return results

def format_frequencies(frequencies: dict[Position, float]) -> str:
    """
    Formats positional frequencies in Yang order: L, F, B, C.
    """

    return (
        f"L={frequencies[Position.LEADER]:.3f}, "
        f"F={frequencies[Position.FOLLOWER]:.3f}, "
        f"B={frequencies[Position.BORDER]:.3f}, "
        f"C={frequencies[Position.CENTRE]:.3f}"
    )


def format_strategy(strategy: StrategyPrey) -> str:
    """
    Formats c = (p(F|L), p(L|F), p(L|B)).
    """

    return (
        f"p(F|L)={strategy.p_F_given_L:.3f}, "
        f"p(L|F)={strategy.p_L_given_F:.3f}, "
        f"p(L|B)={strategy.p_L_given_B:.3f}"
    )


def format_figure_reproduction_result(
    result: FigureReproductionResult,
) -> str:
    """
    Creates a readable report for one reproduction target.
    """

    lines = [
        f"=== {result.target.name} ===",
        f"Environment X: ({result.target.environment.X_L:.2f}, "
        f"{result.target.environment.X_F:.2f}, "
        f"{result.target.environment.X_B:.2f}, "
        f"{result.target.environment.X_C:.2f})",
        f"Observed strategy: {format_strategy(result.observed_strategy)}",
        f"Expected strategy: {format_strategy(result.target.expected_strategy)}",
        f"Mean strategy absolute error: {result.mean_strategy_absolute_error:.4f}",
        f"Observed frequencies: {format_frequencies(result.observed_frequencies)}",
        f"Expected frequencies: {format_frequencies(result.target.expected_frequencies)}",
        f"Mean frequency absolute error: {result.mean_frequency_absolute_error:.4f}",
        f"Observed CM: {result.observed_cm:.4f}",
    ]

    if result.target.expected_cm is not None:
        lines.append(f"Expected CM: {result.target.expected_cm:.4f}")
        lines.append(f"CM absolute error: {result.cm_absolute_error:.4f}")

    if result.late_generation_average is not None:
        lines.append(
            "Late-generation average used: "
            f"generations {result.late_generation_average.start_generation}"
            f"-{result.late_generation_average.end_generation} "
            f"({result.late_generation_average.number_of_records} records)"
        )

    return "\n".join(lines)


def print_figure_reproduction_result(
    result: FigureReproductionResult,
) -> None:
    """
    Prints one formatted reproduction result.
    """

    print(format_figure_reproduction_result(result))


def print_figure_reproduction_report(
    results: list[FigureReproductionResult],
) -> None:
    """
    Prints all reproduction results.
    """

    for result in results:
        print_figure_reproduction_result(result)
        print()



def save_to_csv(
    result: FigureReproductionResult,
    csv_path: str,
) -> None:
    """
    Saves generation-by-generation GA records for one figure target.

    This is the important CSV for Yang-style plots because it stores:
    - generation
    - strategy c = (p(F|L), p(L|F), p(L|B))
    - positional frequencies f = (f_L, f_F, f_B, f_C)
    - collective mobility CM
    - risk and fitness

    It does not rerun anything. It only saves the records from an already completed run.
    """

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target = result.target
    records = result.ga_result.records

    with path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "target_name",
            "generation",

            "X_L",
            "X_F",
            "X_B",
            "X_C",

            "mean_risk",
            "mean_fitness",

            "p_F_given_L",
            "p_L_given_F",
            "p_L_given_B",

            "f_L",
            "f_F",
            "f_B",
            "f_C",

            "CM",

            "expected_p_F_given_L",
            "expected_p_L_given_F",
            "expected_p_L_given_B",

            "expected_f_L",
            "expected_f_F",
            "expected_f_B",
            "expected_f_C",

            "expected_CM",
        ])

        for record in records:
            writer.writerow([
                target.name,
                record.generation,

                target.environment.X_L,
                target.environment.X_F,
                target.environment.X_B,
                target.environment.X_C,

                record.mean_risk,
                record.mean_fitness,

                record.mean_strategy.p_F_given_L,
                record.mean_strategy.p_L_given_F,
                record.mean_strategy.p_L_given_B,

                record.average_frequencies[Position.LEADER],
                record.average_frequencies[Position.FOLLOWER],
                record.average_frequencies[Position.BORDER],
                record.average_frequencies[Position.CENTRE],

                record.collective_mobility,

                target.expected_strategy.p_F_given_L,
                target.expected_strategy.p_L_given_F,
                target.expected_strategy.p_L_given_B,

                target.expected_frequencies[Position.LEADER],
                target.expected_frequencies[Position.FOLLOWER],
                target.expected_frequencies[Position.BORDER],
                target.expected_frequencies[Position.CENTRE],

                target.expected_cm,
            ])


def save_summary_to_csv(
    results: list[FigureReproductionResult],
    csv_path: str,
) -> None:
    """
    Saves one final summary row per figure target.
    Useful for bar plots and final comparison tables.
    """

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "target_name",

            "X_L",
            "X_F",
            "X_B",
            "X_C",

            "observed_p_F_given_L",
            "observed_p_L_given_F",
            "observed_p_L_given_B",

            "expected_p_F_given_L",
            "expected_p_L_given_F",
            "expected_p_L_given_B",

            "observed_f_L",
            "observed_f_F",
            "observed_f_B",
            "observed_f_C",

            "expected_f_L",
            "expected_f_F",
            "expected_f_B",
            "expected_f_C",

            "observed_CM",
            "expected_CM",

            "mean_strategy_absolute_error",
            "mean_frequency_absolute_error",
            "cm_absolute_error",

            "late_average_used",
            "late_average_start_generation",
            "late_average_end_generation",
        ])

        for result in results:
            late_average = result.late_generation_average

            writer.writerow([
                result.target.name,

                result.target.environment.X_L,
                result.target.environment.X_F,
                result.target.environment.X_B,
                result.target.environment.X_C,

                result.observed_strategy.p_F_given_L,
                result.observed_strategy.p_L_given_F,
                result.observed_strategy.p_L_given_B,

                result.target.expected_strategy.p_F_given_L,
                result.target.expected_strategy.p_L_given_F,
                result.target.expected_strategy.p_L_given_B,

                result.observed_frequencies[Position.LEADER],
                result.observed_frequencies[Position.FOLLOWER],
                result.observed_frequencies[Position.BORDER],
                result.observed_frequencies[Position.CENTRE],

                result.target.expected_frequencies[Position.LEADER],
                result.target.expected_frequencies[Position.FOLLOWER],
                result.target.expected_frequencies[Position.BORDER],
                result.target.expected_frequencies[Position.CENTRE],

                result.observed_cm,
                result.target.expected_cm,

                result.mean_strategy_absolute_error,
                result.mean_frequency_absolute_error,
                result.cm_absolute_error,

                late_average is not None,
                late_average.start_generation if late_average is not None else "",
                late_average.end_generation if late_average is not None else "",
            ])


if __name__ == "__main__":
    PRACTICAL_SETTINGS = {
        "population_size": 200,
        "generations": 500,
        "episode_steps": 500,
        "replacement_rate": 0.10,
        "tournament_size": 4,
        "mutation_std": 0.02,
        "r_F_per_L": 8,
        "relocation_interval": 100,
        "record_interval": 10,
    }

    output_dir = Path("yang_reproduction_outputs")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for index, target in enumerate(FIGURE_REPRODUCTION_TARGETS):
        print(f"\nRunning {target.name}...")

        result = reproduce_figure_target(
            target=target,
            seed=42 + index,
            **PRACTICAL_SETTINGS,
        )

        print_figure_reproduction_result(result)

        safe_target_name = target.name.lower().replace(" ", "_")

        records_csv_path = output_dir / f"{safe_target_name}_generation_records.csv"

        save_to_csv(
            result=result,
            csv_path=str(records_csv_path),
        )

        print(f"Saved generation records to: {records_csv_path}")

        all_results.append(result)

    summary_csv_path = output_dir / "figure_reproduction_summary.csv"

    save_summary_to_csv(
        results=all_results,
        csv_path=str(summary_csv_path),
    )

    print(f"\nSaved final summary to: {summary_csv_path}")