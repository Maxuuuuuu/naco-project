import csv
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

    step = 0.0025 #0.1, 0.05, 0.25, 0.01, 0.0025
    out_file = "results_step_0_1.csv"

    rows = []
    for idx, env in enumerate(generate_fixed_environments(step=step), start=1):
        frequencies, fitness = evaluate_strategy(
            strategy=strategy,
            initial_state=initial_state,
            environment=env,
            steps=500,
            burn_in=100,
            r_F_L=8,
            relocation_interval=200,
        )

        rows.append({
            "env_id": idx,
            "X_L": env.X_L,
            "X_F": env.X_F,
            "X_B": env.X_B,
            "X_C": env.X_C,
            "f_L": frequencies["L"],
            "f_F": frequencies["F"],
            "f_B": frequencies["B"],
            "f_C": frequencies["C"],
            "fitness": fitness,
        })

        print(
            f"[{idx}] X=({env.X_L:.2f}, {env.X_F:.2f}, {env.X_B:.2f}, {env.X_C:.2f}) "
            f"-> fitness={fitness:.6f}"
        )

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()