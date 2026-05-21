from genetic_algorithm import generate_prey
from prototype_dev import (
    Strategy_prey,
    Population_state,
    Env,
    update_population_once,
    evaluate_strategy,
)

# --- basic setup ---
#strategy = Strategy_prey(0.8, 0.2, 0.6)

initial_state = Population_state(10, 30, 40, 20)

strategies = generate_prey(initial_state)

environment = Env(0.1, 0.1, 0.7, 0.1)

# --- Test 1: population size stays constant ---
state = initial_state
for _ in range(1000):
    state = update_population_once(strategies)
    assert state.total == initial_state.total

print("Population size test passed.")

# --- Test 2: frequencies sum to 1 ---
# f, fitness = evaluate_strategy(strategy, initial_state, environment)
#
# assert abs(sum(f.values()) - 1.0) < 1e-9
#
# print("Frequencies:", f)
# print("Fitness:", fitness)
# print("Frequency sum test passed.")