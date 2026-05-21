import csv

import numpy as np

from experiment import Experiment
from genetic_algorithm import generate_prey, create_generation
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

experiments: list[Experiment] = [
    Experiment("1A", Env(0.18, 0.04, 0.53, 0.25)),
    Experiment("1B", Env(0.45, 0.22, 0.24, 0.09)),
    Experiment("2A", Env(0.38, 0.15, 0.11, 0.36)),
    Experiment("2B", Env(0.25, 0.19, 0.03, 0.53))
]


def main():
    generations = 500
    #
    # step = 0.25 #0.1, 0.05, 0.25, 0.01, 0.0025
    # For stochasticity.
    runs_per_env = 3
    for exp in experiments:
        out_file = f"results/test/{exp.name}.csv"
        rows = []

        for run in range(runs_per_env):
            print(f"Running experiment: {exp.name} - {run}")
            strategies = generate_prey(generations)
            initial_state = get_state_counts(strategies)

            for gen in range(generations):
                if gen % 100 == 0:
                    print(f"Generation: {gen}")
                frequencies, fitness = evaluate_strategy(
                    strategies=strategies,
                    initial_state=initial_state,
                    environment=exp.env,
                    steps=500,
                    burn_in=100,
                    r_F_L=8,
                    relocation_interval=100,
                )

                rows.append({
                    "target_name": exp.name,
                    "run": run,
                    "gen": gen,
                    "X_L": exp.env.X_L,
                    "X_F": exp.env.X_F,
                    "X_B": exp.env.X_B,
                    "X_C": exp.env.X_C,
                    "f_L": frequencies["L"],
                    "f_F": frequencies["F"],
                    "f_B": frequencies["B"],
                    "f_C": frequencies["C"],
                    "mean_fitness": np.mean([f[1] for f in fitness]),
                    "mean_p_F_given_L": np.mean([f[0].p_F_given_L for f in fitness]),
                    "mean_p_L_given_F": np.mean([f[0].p_L_given_F for f in fitness]),
                    "mean_p_L_given_B": np.mean([f[0].p_L_given_B for f in fitness]),
                    "CM": frequencies["L"] + frequencies["F"]
                })

                strategies = create_generation(fitness)

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()