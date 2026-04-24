from prototype_dev import Strategy_prey, Env, Population_state, update_population_once, evaluate_strategy

strategy = Strategy_prey(
    p_F_given_L=0.8,
    p_L_given_F=0.2,
    p_L_given_B=0.6,
)

strategy.validation()

env = Env(
    X_L=4,
    X_F=1,
    X_B=3,
    X_C=2,
).normalize()

print(env)
print(env.vector())


state = Population_state(n_L=10, n_F=30, n_B=40, n_C=20)

strategy = Strategy_prey(
    p_F_given_L=0.8,
    p_L_given_F=0.2,
    p_L_given_B=0.6,
)

for _ in range(5):
    state = update_population_once(state, strategy)
    print(state)

state = Population_state(n_L=1, n_F=90, n_B=5, n_C=4)

strategy = Strategy_prey(
    p_F_given_L=0.8,
    p_L_given_F=0.2,
    p_L_given_B=0.6,
)

for _ in range(5):
    state = update_population_once(state, strategy)
    print(state, "total =", state.total)


strategy = Strategy_prey(
    p_F_given_L=0.8,
    p_L_given_F=0.2,
    p_L_given_B=0.6,
)

initial_state = Population_state(
    n_L=10,
    n_F=30,
    n_B=40,
    n_C=20,
)

environment = Env(
    X_F=0.4,
    X_L=0.1,
    X_B=0.3,
    X_C=0.2,
)

frequencies, fitness = evaluate_strategy(
    strategy=strategy,
    initial_state=initial_state,
    environment=environment,
    steps=500,
    burn_in=100,
)

print("Frequencies:", frequencies)
print("Fitness:", fitness)
print("Frequency sum:", sum(frequencies.values()))