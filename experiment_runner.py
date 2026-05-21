import csv

from genetic_algorithm import generate_prey, create_generation
from prototype_dev import Strategy_prey, Env, Population_state, evaluate_strategy

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


def main():
    generations = 200
    initial_state = Population_state(10, 30, 40, 20)

    step = 0.25 #0.1, 0.05, 0.25, 0.01, 0.0025
    out_file = "results_step_0_1.csv"

    rows = []
    steps = generate_fixed_environments(step=step)
    print(f"Compute {steps} environments")

    for idx, env in enumerate(steps, start=1):
        print(f"Running environment: {idx}")
        print(env)
        strategies = generate_prey(initial_state)
        for gen in range(generations):
            frequencies, fitness = evaluate_strategy(
                strategies=strategies,
                initial_state=initial_state,
                environment=env,
                steps=600,
                burn_in=100,
                r_F_L=8,
                relocation_interval=200,
            )

            rows.append({
                "env_id": idx,
                "gen": gen,
                "X_L": env.X_L,
                "X_F": env.X_F,
                "X_B": env.X_B,
                "X_C": env.X_C,
                "f_L": frequencies["L"],
                "f_F": frequencies["F"],
                "f_B": frequencies["B"],
                "f_C": frequencies["C"],
                #"avg_fitness": sum(fitness) / len(fitness),
            })

            strategies = create_generation(fitness)

        # print(
        #     f"[{idx}] X=({env.X_L:.2f}, {env.X_F:.2f}, {env.X_B:.2f}, {env.X_C:.2f}) "
        #     f"-> fitness={fitness:.6f}"
        # )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()