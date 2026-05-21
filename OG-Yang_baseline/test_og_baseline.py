from yang_og_baseline import (
    StrategyPrey,
    Environment,
    Position,
    PreyAgent,
    count_positions,
    position_frequencies,
    enforce_mobile_group_constraint,
    calculate_target_centres,
    enforce_herd_geometry_constraint,
    enforce_geometric_constraints,
    random_position,
    create_random_population,
    run_position_episode,
    average_individual_frequencies,
    compute_individual_risk,
    compute_individual_fitness,
    compute_population_risks,
    compute_population_fitnesses,
    compute_mean_risk,
    clone_strategy,
    mutate_strategy,
    calculate_replacement_count,
    tournament_select_parent,
    get_replacement_indices,
    ga_search_update,
    mean_strategy,
    ESSComparison,
    create_population_from_strategy,
    create_perturbed_population,
    mean,
    values_for_label,
    frequencies_for_label,
    evaluate_resident_against_mutant,
    evaluate_resident_against_many_mutants,
    is_stable_against_all_mutants,
    randomize_population_positions,
    GenerationRecord,
    GASearchResult,
    compute_collective_mobility,
    create_generation_record,
    run_ga_search,    
)

# Phase 1 tests
def test_strategy_clip():
    strategy = StrategyPrey(
        p_F_given_L=1.2,
        p_L_given_F=-0.3,
        p_L_given_B=0.5,
    )

    strategy.clip()

    assert strategy.p_F_given_L == 1.0
    assert strategy.p_L_given_F == 0.0
    assert strategy.p_L_given_B == 0.5


def test_environment_normalization():
    env = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    normalized = env.normalize()
    total = normalized.X_L + normalized.X_F + normalized.X_B + normalized.X_C

    assert abs(total - 1.0) < 1e-9


def test_environment_dict():
    env = Environment(0.18, 0.04, 0.53, 0.25)
    env_dict = env.as_dict()

    assert Position.LEADER in env_dict
    assert Position.FOLLOWER in env_dict
    assert Position.BORDER in env_dict
    assert Position.CENTRE in env_dict

# Phase 2 tests
def test_L_becomes_F_when_probablility_is_1():
    strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.LEADER)
    agent.update_position()

    assert agent.position == Position.FOLLOWER

def test_L_stays_L_when_probablility_is_0():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.LEADER)
    agent.update_position()

    assert agent.position == Position.LEADER

def test_F_becomes_L_when_probablility_is_1():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=1.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.FOLLOWER)
    agent.update_position()

    assert agent.position == Position.LEADER

def test_F_stays_F_when_probablility_is_0():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.FOLLOWER)
    agent.update_position()

    assert agent.position == Position.FOLLOWER

def test_B_becomes_L_when_probablility_is_1():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=1.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.BORDER)
    agent.update_position()

    assert agent.position == Position.LEADER

def test_B_stays_B_when_probablility_is_0():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.BORDER)
    agent.update_position()

    assert agent.position == Position.BORDER

def test_C_stays_C():
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    agent = PreyAgent(strategy=strategy, position=Position.CENTRE)
    agent.update_position()

    assert agent.position == Position.CENTRE


# Phase 3 tests
def make_prey(position: Position) -> PreyAgent:
    strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=0.0,
        p_L_given_B=0.0,
    )

    return PreyAgent(
        strategy=strategy,
        position=position,
    )


def test_count_positions():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
        make_prey(Position.CENTRE),
    ]

    counts = count_positions(population)

    assert counts[Position.LEADER] == 1
    assert counts[Position.FOLLOWER] == 2
    assert counts[Position.BORDER] == 1
    assert counts[Position.CENTRE] == 1


def test_position_frequencies_sum_to_one():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
        make_prey(Position.CENTRE),
    ]

    frequencies = position_frequencies(population)
    total = sum(frequencies.values())

    assert abs(total - 1.0) < 1e-9


def test_mobile_constraint_moves_excess_followers_to_border():
    population = []

    # 1 leader allows at most 8 followers.
    population.append(make_prey(Position.LEADER))

    # 10 followers means 2 should become border.
    for _ in range(10):
        population.append(make_prey(Position.FOLLOWER))

    enforce_mobile_group_constraint(population, r_F_per_L=8)

    counts = count_positions(population)

    assert counts[Position.LEADER] == 1
    assert counts[Position.FOLLOWER] == 8
    assert counts[Position.BORDER] == 2
    assert len(population) == 11


def test_mobile_constraint_with_no_leaders_moves_all_followers_to_border():
    population = [
        make_prey(Position.FOLLOWER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.FOLLOWER),
    ]

    enforce_mobile_group_constraint(population, r_F_per_L=8)

    counts = count_positions(population)

    assert counts[Position.LEADER] == 0
    assert counts[Position.FOLLOWER] == 0
    assert counts[Position.BORDER] == 3
    assert len(population) == 3


def test_calculate_target_centres_for_small_herd():
    assert calculate_target_centres(0) == 0
    assert calculate_target_centres(3) == 0
    assert calculate_target_centres(7) == 0


def test_calculate_target_centres_for_larger_herd():
    # Cube-like intuition:
    # 27 individuals -> 3 x 3 x 3 cube
    # inner centre block is roughly 1 x 1 x 1 = 1
    assert calculate_target_centres(27) == 1


def test_herd_geometry_constraint_preserves_population_size():
    population = []

    for _ in range(30):
        population.append(make_prey(Position.BORDER))

    before_size = len(population)

    enforce_herd_geometry_constraint(population)

    after_size = len(population)
    counts = count_positions(population)

    assert before_size == after_size
    assert counts[Position.BORDER] + counts[Position.CENTRE] == 30


def test_full_geometric_constraints_preserve_population_size():
    population = []

    for _ in range(2):
        population.append(make_prey(Position.LEADER))

    for _ in range(30):
        population.append(make_prey(Position.FOLLOWER))

    for _ in range(20):
        population.append(make_prey(Position.BORDER))

    before_size = len(population)

    enforce_geometric_constraints(population, r_F_per_L=8)

    after_size = len(population)
    counts = count_positions(population)

    assert before_size == after_size
    assert counts[Position.FOLLOWER] <= 8 * counts[Position.LEADER]
    assert sum(counts.values()) == before_size


# Phase 4 tests
def test_random_position_returns_valid_position():
    position = random_position()

    assert position in {
        Position.LEADER,
        Position.FOLLOWER,
        Position.BORDER,
        Position.CENTRE,
    }


def test_create_random_population_has_correct_size():
    population = create_random_population(population_size=25)

    assert len(population) == 25

    for prey in population:
        assert isinstance(prey, PreyAgent)
        assert prey.position in {
            Position.LEADER,
            Position.FOLLOWER,
            Position.BORDER,
            Position.CENTRE,
        }

        assert 0.0 <= prey.strategy.p_F_given_L <= 1.0
        assert 0.0 <= prey.strategy.p_L_given_F <= 1.0
        assert 0.0 <= prey.strategy.p_L_given_B <= 1.0


def test_run_position_episode_returns_frequency_for_each_prey():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
        make_prey(Position.CENTRE),
    ]

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
    )

    assert len(individual_frequencies) == len(population)


def test_each_individual_frequency_sums_to_one():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
        make_prey(Position.CENTRE),
    ]

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
    )

    for frequencies in individual_frequencies:
        total = sum(frequencies.values())
        assert abs(total - 1.0) < 1e-9


def test_average_individual_frequencies_sum_to_one():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
        make_prey(Position.CENTRE),
    ]

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
    )

    average_frequencies = average_individual_frequencies(
        individual_frequencies
    )

    total = sum(average_frequencies.values())

    assert abs(total - 1.0) < 1e-9


def test_run_position_episode_preserves_population_size():
    population = create_random_population(population_size=50)

    before_size = len(population)

    run_position_episode(
        population=population,
        steps=20,
        r_F_per_L=8,
    )

    after_size = len(population)

    assert before_size == after_size


# Phase 5 tests
def test_compute_individual_risk_for_pure_leader():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    frequencies = {
        Position.LEADER: 1.0,
        Position.FOLLOWER: 0.0,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    risk = compute_individual_risk(
        frequencies=frequencies,
        environment=environment,
    )

    assert abs(risk - 0.18) < 1e-9


def test_compute_individual_risk_for_mixed_frequencies():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    frequencies = {
        Position.LEADER: 0.5,
        Position.FOLLOWER: 0.5,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    risk = compute_individual_risk(
        frequencies=frequencies,
        environment=environment,
    )

    expected_risk = 0.18 * 0.5 + 0.04 * 0.5

    assert abs(risk - expected_risk) < 1e-9


def test_fitness_is_negative_risk():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    frequencies = {
        Position.LEADER: 1.0,
        Position.FOLLOWER: 0.0,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    risk = compute_individual_risk(
        frequencies=frequencies,
        environment=environment,
    )

    fitness = compute_individual_fitness(
        frequencies=frequencies,
        environment=environment,
    )

    assert fitness == -risk


def test_population_risks_length_matches_population():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    individual_frequencies = [
        {
            Position.LEADER: 1.0,
            Position.FOLLOWER: 0.0,
            Position.BORDER: 0.0,
            Position.CENTRE: 0.0,
        },
        {
            Position.LEADER: 0.0,
            Position.FOLLOWER: 1.0,
            Position.BORDER: 0.0,
            Position.CENTRE: 0.0,
        },
    ]

    risks = compute_population_risks(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    assert len(risks) == 2
    assert abs(risks[0] - 0.18) < 1e-9
    assert abs(risks[1] - 0.04) < 1e-9


def test_mean_risk():
    risks = [0.1, 0.2, 0.3]

    mean_risk = compute_mean_risk(risks)

    assert abs(mean_risk - 0.2) < 1e-9

def test_population_fitnesses_are_negative_risks():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    individual_frequencies = [
        {
            Position.LEADER: 1.0,
            Position.FOLLOWER: 0.0,
            Position.BORDER: 0.0,
            Position.CENTRE: 0.0,
        },
        {
            Position.LEADER: 0.0,
            Position.FOLLOWER: 1.0,
            Position.BORDER: 0.0,
            Position.CENTRE: 0.0,
        },
    ]

    risks = compute_population_risks(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    fitnesses = compute_population_fitnesses(
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    assert len(fitnesses) == 2
    assert fitnesses[0] == -risks[0]
    assert fitnesses[1] == -risks[1]
    assert abs(fitnesses[0] - (-0.18)) < 1e-9
    assert abs(fitnesses[1] - (-0.04)) < 1e-9

# Phase 6 tests
def test_clone_strategy_copies_values():
    strategy = StrategyPrey(
        p_F_given_L=0.1,
        p_L_given_F=0.2,
        p_L_given_B=0.3,
    )

    copied = clone_strategy(strategy)

    assert copied.p_F_given_L == 0.1
    assert copied.p_L_given_F == 0.2
    assert copied.p_L_given_B == 0.3
    assert copied is not strategy


def test_mutate_strategy_keeps_values_between_zero_and_one():
    strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.0,
        p_L_given_B=0.5,
    )

    mutated = mutate_strategy(
        strategy=strategy,
        mutation_std=1.0,
    )

    assert 0.0 <= mutated.p_F_given_L <= 1.0
    assert 0.0 <= mutated.p_L_given_F <= 1.0
    assert 0.0 <= mutated.p_L_given_B <= 1.0


def test_calculate_replacement_count():
    assert calculate_replacement_count(1000, 0.10) == 100
    assert calculate_replacement_count(10, 0.10) == 1
    assert calculate_replacement_count(25, 0.10) == 2


def test_tournament_select_parent_selects_best_when_all_are_candidates():
    population = [
        make_prey(Position.LEADER),
        make_prey(Position.FOLLOWER),
        make_prey(Position.BORDER),
    ]

    fitnesses = [-0.3, -0.1, -0.2]

    parent = tournament_select_parent(
        population=population,
        fitnesses=fitnesses,
        tournament_size=3,
    )

    assert parent is population[1]


def test_get_replacement_indices_returns_lowest_fitness_indices():
    fitnesses = [-0.3, -0.1, -0.5, -0.2]

    replacement_indices = get_replacement_indices(
        fitnesses=fitnesses,
        replacement_count=2,
    )

    assert replacement_indices == [2, 0]


def test_ga_search_update_preserves_population_size():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    population = create_random_population(20)

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
    )

    before_size = len(population)

    ga_search_update(
        population=population,
        individual_frequencies=individual_frequencies,
        environment=environment,
        replacement_rate=0.10,
        tournament_size=4,
        mutation_std=0.02,
    )

    after_size = len(population)

    assert before_size == after_size


def test_mean_strategy_returns_valid_probabilities():
    population = create_random_population(20)

    strategy = mean_strategy(population)

    assert 0.0 <= strategy.p_F_given_L <= 1.0
    assert 0.0 <= strategy.p_L_given_F <= 1.0
    assert 0.0 <= strategy.p_L_given_B <= 1.0

def test_create_population_from_strategy_has_correct_size_and_strategy():
    strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.01,
        p_L_given_B=1.0,
    )

    population = create_population_from_strategy(
        strategy=strategy,
        population_size=10,
    )

    assert len(population) == 10

    for prey in population:
        assert prey.strategy.p_F_given_L == 1.0
        assert prey.strategy.p_L_given_F == 0.01
        assert prey.strategy.p_L_given_B == 1.0
        assert prey.strategy is not strategy


def test_create_perturbed_population_counts_residents_and_mutants():
    resident_strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.01,
        p_L_given_B=1.0,
    )

    mutant_strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=1.0,
        p_L_given_B=0.0,
    )

    population, labels = create_perturbed_population(
        resident_strategy=resident_strategy,
        mutant_strategy=mutant_strategy,
        population_size=100,
        mutant_fraction=0.05,
    )

    assert len(population) == 100
    assert len(labels) == 100
    assert labels.count("resident") == 95
    assert labels.count("mutant") == 5


def test_mean_function():
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_values_for_label():
    values = [0.1, 0.2, 0.3, 0.4]
    labels = ["resident", "mutant", "resident", "mutant"]

    resident_values = values_for_label(
        values=values,
        labels=labels,
        target_label="resident",
    )

    mutant_values = values_for_label(
        values=values,
        labels=labels,
        target_label="mutant",
    )

    assert resident_values == [0.1, 0.3]
    assert mutant_values == [0.2, 0.4]


def test_frequencies_for_label():
    frequency_a = {
        Position.LEADER: 1.0,
        Position.FOLLOWER: 0.0,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    frequency_b = {
        Position.LEADER: 0.0,
        Position.FOLLOWER: 1.0,
        Position.BORDER: 0.0,
        Position.CENTRE: 0.0,
    }

    individual_frequencies = [frequency_a, frequency_b]
    labels = ["resident", "mutant"]

    resident_frequencies = frequencies_for_label(
        individual_frequencies=individual_frequencies,
        labels=labels,
        target_label="resident",
    )

    mutant_frequencies = frequencies_for_label(
        individual_frequencies=individual_frequencies,
        labels=labels,
        target_label="mutant",
    )

    assert resident_frequencies == [frequency_a]
    assert mutant_frequencies == [frequency_b]


def test_evaluate_resident_against_mutant_returns_ess_comparison():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    resident_strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.01,
        p_L_given_B=1.0,
    )

    mutant_strategy = StrategyPrey(
        p_F_given_L=0.0,
        p_L_given_F=1.0,
        p_L_given_B=0.0,
    )

    comparison = evaluate_resident_against_mutant(
        resident_strategy=resident_strategy,
        mutant_strategy=mutant_strategy,
        environment=environment,
        population_size=50,
        mutant_fraction=0.10,
        steps=20,
        r_F_per_L=8,
    )

    assert isinstance(comparison, ESSComparison)
    assert comparison.resident_count == 45
    assert comparison.mutant_count == 5

    assert isinstance(comparison.is_resident_stable, bool)

    assert abs(
        sum(comparison.resident_average_frequencies.values()) - 1.0
    ) < 1e-9

    assert abs(
        sum(comparison.mutant_average_frequencies.values()) - 1.0
    ) < 1e-9

    assert abs(
        sum(comparison.overall_average_frequencies.values()) - 1.0
    ) < 1e-9


def test_evaluate_resident_against_many_mutants_returns_list():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    resident_strategy = StrategyPrey(
        p_F_given_L=1.0,
        p_L_given_F=0.01,
        p_L_given_B=1.0,
    )

    mutant_strategies = [
        StrategyPrey(
            p_F_given_L=0.0,
            p_L_given_F=1.0,
            p_L_given_B=0.0,
        ),
        StrategyPrey(
            p_F_given_L=0.5,
            p_L_given_F=0.5,
            p_L_given_B=0.5,
        ),
    ]

    comparisons = evaluate_resident_against_many_mutants(
        resident_strategy=resident_strategy,
        mutant_strategies=mutant_strategies,
        environment=environment,
        population_size=50,
        mutant_fraction=0.10,
        steps=20,
        r_F_per_L=8,
    )

    assert len(comparisons) == 2

    for comparison in comparisons:
        assert isinstance(comparison, ESSComparison)


def test_is_stable_against_all_mutants():
    comparison_1 = ESSComparison(
        resident_strategy=StrategyPrey(1.0, 0.01, 1.0),
        mutant_strategy=StrategyPrey(0.0, 1.0, 0.0),
        resident_count=95,
        mutant_count=5,
        resident_mean_risk=0.10,
        mutant_mean_risk=0.20,
        resident_mean_fitness=-0.10,
        mutant_mean_fitness=-0.20,
        resident_average_frequencies={
            Position.LEADER: 0.2,
            Position.FOLLOWER: 0.6,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        mutant_average_frequencies={
            Position.LEADER: 0.4,
            Position.FOLLOWER: 0.4,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        overall_average_frequencies={
            Position.LEADER: 0.21,
            Position.FOLLOWER: 0.59,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        is_resident_stable=True,
    )

    comparison_2 = ESSComparison(
        resident_strategy=StrategyPrey(1.0, 0.01, 1.0),
        mutant_strategy=StrategyPrey(0.5, 0.5, 0.5),
        resident_count=95,
        mutant_count=5,
        resident_mean_risk=0.11,
        mutant_mean_risk=0.21,
        resident_mean_fitness=-0.11,
        mutant_mean_fitness=-0.21,
        resident_average_frequencies={
            Position.LEADER: 0.2,
            Position.FOLLOWER: 0.6,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        mutant_average_frequencies={
            Position.LEADER: 0.4,
            Position.FOLLOWER: 0.4,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        overall_average_frequencies={
            Position.LEADER: 0.21,
            Position.FOLLOWER: 0.59,
            Position.BORDER: 0.1,
            Position.CENTRE: 0.1,
        },
        is_resident_stable=True,
    )

    assert is_stable_against_all_mutants([comparison_1, comparison_2]) is True

    comparison_2.is_resident_stable = False

    assert is_stable_against_all_mutants([comparison_1, comparison_2]) is False

def test_randomize_population_positions_keeps_population_size():
    population = create_random_population(20)

    before_size = len(population)

    randomize_population_positions(population)

    after_size = len(population)

    assert before_size == after_size

    for prey in population:
        assert prey.position in {
            Position.LEADER,
            Position.FOLLOWER,
            Position.BORDER,
            Position.CENTRE,
        }


def test_run_position_episode_with_relocation_interval():
    population = create_random_population(20)

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
        relocation_interval=5,
    )

    assert len(individual_frequencies) == 20

    for frequencies in individual_frequencies:
        assert abs(sum(frequencies.values()) - 1.0) < 1e-9


def test_compute_collective_mobility():
    frequencies = {
        Position.LEADER: 0.2,
        Position.FOLLOWER: 0.5,
        Position.BORDER: 0.2,
        Position.CENTRE: 0.1,
    }

    cm = compute_collective_mobility(frequencies)

    assert abs(cm - 0.7) < 1e-9


def test_create_generation_record_returns_record():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    population = create_random_population(20)

    individual_frequencies = run_position_episode(
        population=population,
        steps=10,
        r_F_per_L=8,
    )

    record = create_generation_record(
        generation=0,
        population=population,
        individual_frequencies=individual_frequencies,
        environment=environment,
    )

    assert isinstance(record, GenerationRecord)
    assert record.generation == 0
    assert 0.0 <= record.collective_mobility <= 1.0
    assert abs(sum(record.average_frequencies.values()) - 1.0) < 1e-9
    assert 0.0 <= record.mean_strategy.p_F_given_L <= 1.0
    assert 0.0 <= record.mean_strategy.p_L_given_F <= 1.0
    assert 0.0 <= record.mean_strategy.p_L_given_B <= 1.0


def test_run_ga_search_returns_result():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    result = run_ga_search(
        environment=environment,
        population_size=30,
        generations=3,
        episode_steps=10,
        replacement_rate=0.10,
        tournament_size=4,
        mutation_std=0.02,
        r_F_per_L=8,
        relocation_interval=5,
        record_interval=1,
        seed=42,
    )

    assert isinstance(result, GASearchResult)
    assert result.population_size == 30
    assert result.generations == 3
    assert result.episode_steps == 10
    assert len(result.records) == 3
    assert len(result.final_population) == 30

    assert 0.0 <= result.final_strategy.p_F_given_L <= 1.0
    assert 0.0 <= result.final_strategy.p_L_given_F <= 1.0
    assert 0.0 <= result.final_strategy.p_L_given_B <= 1.0

    assert abs(sum(result.final_average_frequencies.values()) - 1.0) < 1e-9


def test_run_ga_search_with_record_interval():
    environment = Environment(
        X_L=0.18,
        X_F=0.04,
        X_B=0.53,
        X_C=0.25,
    )

    result = run_ga_search(
        environment=environment,
        population_size=30,
        generations=5,
        episode_steps=10,
        replacement_rate=0.10,
        tournament_size=4,
        mutation_std=0.02,
        r_F_per_L=8,
        relocation_interval=None,
        record_interval=2,
        seed=42,
    )

    recorded_generations = [
        record.generation for record in result.records
    ]

    assert recorded_generations == [0, 2, 4]