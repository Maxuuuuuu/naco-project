import pytest

from yang_og_baseline import (
    Environment,
    FIGURE_1A_TARGET,
    FIGURE_1B_TARGET,
    FIGURE_2A_TARGET,
    FIGURE_2B_TARGET,
    FIGURE_REPRODUCTION_TARGETS,
    FIGURE_REPRODUCTION_SMALL_SETTINGS,
    GenerationRecord,
    Position,
    StrategyPrey,
    average_late_generation_records,
    compute_collective_mobility,
    format_figure_reproduction_result,
    frequency_absolute_errors,
    reproduce_figure_target,
    run_figure_reproduction_suite,
    strategy_absolute_errors,
)

def test_figure_targets_include_figure_1b():
    assert [target.name for target in FIGURE_REPRODUCTION_TARGETS] == [
        "Figure 1A",
        "Figure 1B",
        "Figure 2A",
        "Figure 2B",
    ]

    assert FIGURE_1B_TARGET.environment == Environment(0.45, 0.22, 0.24, 0.09)
    assert FIGURE_1B_TARGET.expected_strategy == StrategyPrey(1.0, 0.0, 0.0)

    assert FIGURE_1B_TARGET.expected_frequencies[Position.LEADER] == pytest.approx(0.00)
    assert FIGURE_1B_TARGET.expected_frequencies[Position.FOLLOWER] == pytest.approx(0.00)
    assert FIGURE_1B_TARGET.expected_frequencies[Position.BORDER] == pytest.approx(0.49)
    assert FIGURE_1B_TARGET.expected_frequencies[Position.CENTRE] == pytest.approx(0.51)

    assert FIGURE_1B_TARGET.expected_cm == pytest.approx(0.00)
    assert FIGURE_1B_TARGET.use_late_generation_average is False


def test_figure_2_targets_use_late_generation_average_but_figure_1_targets_do_not():
    assert FIGURE_1A_TARGET.use_late_generation_average is False
    assert FIGURE_1B_TARGET.use_late_generation_average is False
    assert FIGURE_2A_TARGET.use_late_generation_average is True
    assert FIGURE_2B_TARGET.use_late_generation_average is True

def test_error_helpers_compute_component_errors():
    observed_strategy = StrategyPrey(0.8, 0.2, 0.4)
    expected_strategy = StrategyPrey(1.0, 0.0, 0.25)

    strategy_errors = strategy_absolute_errors(
        observed=observed_strategy,
        expected=expected_strategy,
    )

    assert strategy_errors["p_F_given_L"] == pytest.approx(0.2)
    assert strategy_errors["p_L_given_F"] == pytest.approx(0.2)
    assert strategy_errors["p_L_given_B"] == pytest.approx(0.15)

    observed_frequencies = {
        Position.LEADER: 0.1,
        Position.FOLLOWER: 0.2,
        Position.BORDER: 0.3,
        Position.CENTRE: 0.4,
    }
    expected_frequencies = {
        Position.LEADER: 0.0,
        Position.FOLLOWER: 0.25,
        Position.BORDER: 0.25,
        Position.CENTRE: 0.5,
    }

    frequency_errors = frequency_absolute_errors(
        observed=observed_frequencies,
        expected=expected_frequencies,
    )

    assert frequency_errors[Position.LEADER] == pytest.approx(0.1)
    assert frequency_errors[Position.FOLLOWER] == pytest.approx(0.05)
    assert frequency_errors[Position.BORDER] == pytest.approx(0.05)
    assert frequency_errors[Position.CENTRE] == pytest.approx(0.1)


def test_average_late_generation_records_uses_last_fraction():
    records = []

    for generation in range(6):
        records.append(
            GenerationRecord(
                generation=generation,
                mean_risk=float(generation),
                mean_fitness=-float(generation),
                mean_strategy=StrategyPrey(
                    p_F_given_L=float(generation) / 10,
                    p_L_given_F=float(generation) / 20,
                    p_L_given_B=float(generation) / 30,
                ),
                average_frequencies={
                    Position.LEADER: 0.1 * generation,
                    Position.FOLLOWER: 0.0,
                    Position.BORDER: 1.0 - 0.1 * generation,
                    Position.CENTRE: 0.0,
                },
                collective_mobility=0.1 * generation,
            )
        )

    late_average = average_late_generation_records(
        records=records,
        late_generation_fraction=1 / 3,
    )

    assert late_average.start_generation == 4
    assert late_average.end_generation == 5
    assert late_average.number_of_records == 2
    assert late_average.mean_strategy.p_F_given_L == pytest.approx(0.45)
    assert late_average.mean_frequencies[Position.LEADER] == pytest.approx(0.45)
    assert late_average.mean_collective_mobility == pytest.approx(0.45)

def test_reproduce_figure_1b_target_runs_with_small_settings():
    result = reproduce_figure_target(
        target=FIGURE_1B_TARGET,
        seed=123,
        **FIGURE_REPRODUCTION_SMALL_SETTINGS,
    )

    assert result.target.name == "Figure 1B"
    assert result.late_generation_average is None
    assert 0.0 <= result.observed_cm <= 1.0
    assert result.cm_absolute_error is not None

    assert set(result.observed_frequencies) == {
        Position.LEADER,
        Position.FOLLOWER,
        Position.BORDER,
        Position.CENTRE,
    }

    assert compute_collective_mobility(result.observed_frequencies) == pytest.approx(
        result.observed_cm
    )

def test_run_figure_reproduction_suite_runs_all_targets_with_small_settings():
    results = run_figure_reproduction_suite(
        settings=FIGURE_REPRODUCTION_SMALL_SETTINGS,
        seed=42,
    )

    assert [result.target.name for result in results] == [
        "Figure 1A",
        "Figure 1B",
        "Figure 2A",
        "Figure 2B",
    ]

    assert results[0].late_generation_average is None
    assert results[1].late_generation_average is None
    assert results[2].late_generation_average is not None
    assert results[3].late_generation_average is not None

    formatted = format_figure_reproduction_result(results[1])

    assert "Figure 1B" in formatted
    assert "Observed strategy" in formatted
    assert "Expected frequencies" in formatted