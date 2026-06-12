import csv

import numpy as np

from experiment import Experiment
from genetic_algorithm import generate_prey, create_generation, generate_predators, create_predator_generation
from predator import Predator
from prototype_dev import Strategy_prey, Env, Population_state, evaluate_strategy, get_state_counts


def generate_fixed_environments(step: float = 0.1):
    n = round(1 / step)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                yield Env(
                    X_L=i * step,
                    X_F=j * step,
                    X_B=k * step,
                    X_C=l * step,
                )

def env_from_predators(predators: list[Predator]):
    X_L, X_F, X_B, X_C = 0.0, 0.0, 0.0, 0.0

    for predator in predators:
        X_L += predator.L_predation
        X_F += predator.F_predation
        X_B += predator.B_predation
        X_C += predator.C_predation

    total = X_L + X_F + X_B + X_C

    X_L /= total
    X_F /= total
    X_B /= total
    X_C /= total

    return Env(X_L=X_L, X_F=X_F, X_B=X_B, X_C=X_C)

experiments: list[Experiment] = [
    Experiment("1A", Env(0.18, 0.04, 0.53, 0.25)),
    Experiment("1B", Env(0.45, 0.22, 0.24, 0.09)),
    Experiment("2A", Env(0.38, 0.15, 0.11, 0.36)),
    Experiment("2B", Env(0.25, 0.19, 0.03, 0.53))
]


def main():
    generations = 1000
    #
    # For stochasticity.
    runs_per_env = 3
    use_coevolution = True
    mu = 0.1
    k = 4
    sigma = 0.02

    # for exp in experiments:
    #     out_file = f"results/g1000r3/{exp.name}.csv"
    #     rows = []
    #
    #     for run in range(runs_per_env):
    #         print(f"Running experiment: {exp.name} - {run}")
    #         strategies = generate_prey(1000)
    #         predators = generate_predators(100)
    #         initial_state = get_state_counts(strategies)
    #
    #
    #         for gen in range(generations):
    #             if gen % 100 == 0:
    #                 print(f"Generation: {gen}")
    #             frequencies, fitness, predator_fitness = evaluate_strategy(
    #                 strategies=strategies,
    #                 initial_state=initial_state,
    #                 environment=exp.env,
    #                 steps=500,
    #                 burn_in=100,
    #                 r_F_L=8,
    #                 relocation_interval=100,
    #                 use_coevolution=False,
    #                 predators=predators,
    #             )
    #
    #             rows.append({
    #                 "target_name": exp.name,
    #                 "run": run,
    #                 "gen": gen,
    #                 "X_L": exp.env.X_L,
    #                 "X_F": exp.env.X_F,
    #                 "X_B": exp.env.X_B,
    #                 "X_C": exp.env.X_C,
    #                 "f_L": frequencies["L"],
    #                 "f_F": frequencies["F"],
    #                 "f_B": frequencies["B"],
    #                 "f_C": frequencies["C"],
    #                 "mean_fitness": np.mean([f[1] for f in fitness]),
    #                 "mean_p_F_given_L": np.mean([f[0].p_F_given_L for f in fitness]),
    #                 "mean_p_L_given_F": np.mean([f[0].p_L_given_F for f in fitness]),
    #                 "mean_p_L_given_B": np.mean([f[0].p_L_given_B for f in fitness]),
    #                 "CM": frequencies["L"] + frequencies["F"],
    #             })
    #
    #             strategies = create_generation(fitness, mu = mu, k = k, sigma = sigma)
    #             # if use_coevolution:
    #             #     predators = create_predator_generation(predator_fitness, mu = mu, k = k, sigma = sigma)
    #
    #     with open(out_file, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    #         writer.writeheader()
    #         writer.writerows(rows)
    #
    #     print(f"\nSaved {len(rows)} rows to {out_file}")

    base_file = f"results/baseline/g1000r4/default.csv"
    coevo_file = f"results/coevolution/g1000r4/default.csv"
    base_rows = []
    coevo_rows = []

    for r in range(4):
        print(f"Running experiment: {r}")
        strategies = generate_prey(1000)
        base_strategies = [p.copy() for p in strategies]
        predators = generate_predators(100)
        initial_state = get_state_counts(strategies)
        env = env_from_predators(predators)

        # Coevo loop
        for gen in range(generations):
            if gen % 100 == 0:
                print(f"Generation: {gen}")
            frequencies, fitness, predator_fitness = evaluate_strategy(
                strategies=strategies,
                initial_state=initial_state,
                environment=env,
                steps=500,
                burn_in=100,
                r_F_L=8,
                relocation_interval=100,
                use_coevolution=True,
                predators=predators,
            )

            coevo_rows.append({
                "target_name": f"Experiment {r}",
                "run": r,
                "gen": gen,
                "X_L": np.mean([p[0].L_predation for p in predator_fitness]),
                "X_F": np.mean([p[0].F_predation for p in predator_fitness]),
                "X_B": np.mean([p[0].B_predation for p in predator_fitness]),
                "X_C": np.mean([p[0].C_predation for p in predator_fitness]),
                "f_L": frequencies["L"],
                "f_F": frequencies["F"],
                "f_B": frequencies["B"],
                "f_C": frequencies["C"],
                "mean_fitness": np.mean([f[1] for f in fitness]),
                "pred_fitness": np.mean([f[1] for f in predator_fitness]),
                "mean_p_F_given_L": np.mean([f[0].p_F_given_L for f in fitness]),
                "mean_p_L_given_F": np.mean([f[0].p_L_given_F for f in fitness]),
                "mean_p_L_given_B": np.mean([f[0].p_L_given_B for f in fitness]),
                "CM": frequencies["L"] + frequencies["F"],
            })

            strategies = create_generation(fitness, mu = mu, k = k, sigma = sigma)
            predators = create_predator_generation(predator_fitness, mu = mu, k = k, sigma = sigma)

        initial_state = get_state_counts(base_strategies)
        # Baseloop
        for gen in range(generations):
            if gen % 100 == 0:
                print(f"Generation: {gen}")
            frequencies, fitness, predator_fitness = evaluate_strategy(
                strategies=base_strategies,
                initial_state=initial_state,
                environment=env,
                steps=500,
                burn_in=100,
                r_F_L=8,
                relocation_interval=100,
                use_coevolution=False,
                predators=[],
            )

            base_rows.append({
                "target_name": f"Experiment {r}",
                "run": r,
                "gen": gen,
                "X_L": env.X_L,
                "X_F": env.X_F,
                "X_B": env.X_B,
                "X_C": env.X_C,
                "f_L": frequencies["L"],
                "f_F": frequencies["F"],
                "f_B": frequencies["B"],
                "f_C": frequencies["C"],
                "prey_fitness": np.mean([f[1] for f in fitness]),
                "pred_fitness": np.mean([f[1] for f in predator_fitness]),
                "mean_p_F_given_L": np.mean([f[0].p_F_given_L for f in fitness]),
                "mean_p_L_given_F": np.mean([f[0].p_L_given_F for f in fitness]),
                "mean_p_L_given_B": np.mean([f[0].p_L_given_B for f in fitness]),
                "CM": frequencies["L"] + frequencies["F"],
            })

            base_strategies = create_generation(fitness, mu = mu, k = k, sigma = sigma)

    with open(coevo_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=coevo_rows[0].keys())
        writer.writeheader()
        writer.writerows(coevo_rows)
    print(f"\nSaved {len(coevo_rows)} rows to {coevo_file}")

    with open(base_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_rows[0].keys())
        writer.writeheader()
        writer.writerows(base_rows)
    print(f"\nSaved {len(base_rows)} rows to {base_file}")


if __name__ == "__main__":
    main()