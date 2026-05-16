"""
Quality checks for APPROX-SUBSET-SUM.

For each instance we compute the exact optimum once, then run the heuristic
for several eps values and record ratios and feasibility.

CSV columns:
    n, trial, seed, epsilon, exact_sum, approx_sum, ratio,
    bound_lower, bound_satisfied, feasible
"""

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from sample_generator import generate_instance
from brute_force import brute_force_subset_sum
from heuristic import approx_subset_sum


def run_trial(n: int, trial: int, seed: int, epsilons: List[float], max_weight: int) -> List[dict]:
    """Generate one instance, compute exact once, then run per epsilon."""
    S, t = generate_instance(n=n, max_weight=max_weight, seed=seed)

    # Exact optimum (independent of epsilon)
    exact_sum, _ = brute_force_subset_sum(S, t)

    rows = []
    for epsilon in epsilons:
        approx_sum = approx_subset_sum(S, t, epsilon)

        # Quality metrics
        if exact_sum > 0:
            ratio = approx_sum / exact_sum
        else:
            ratio = 1.0  # edge case: both are 0; treat as exact match
        bound_lower = 1.0 / (1.0 + epsilon)
        bound_satisfied = ratio >= bound_lower - 1e-12  # tiny float slack
        feasible = (approx_sum <= t) and (approx_sum <= exact_sum)

        rows.append({
            "n": n,
            "trial": trial,
            "seed": seed,
            "epsilon": epsilon,
            "exact_sum": exact_sum,
            "approx_sum": approx_sum,
            "ratio": ratio,
            "bound_lower": bound_lower,
            "bound_satisfied": int(bound_satisfied),
            "feasible": int(feasible),
        })
    return rows


def run_quality_experiment(
    n_values: List[int],
    epsilons: List[float],
    trials_per_n: int = 30,
    max_weight: int = 1000,
    output_csv: str = "data/quality_data.csv",
    seed_base: int = 1_000_000,   # offset from performance seeds so instances differ
    num_workers: int = 8,
) -> None:
    """Run the quality experiment in parallel across (n, trial) pairs."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[int, int, int]] = []
    for n in n_values:
        for trial in range(trials_per_n):
            seed = seed_base + n * 10_000 + trial
            tasks.append((n, trial, seed))

    total_pairs = len(tasks)
    total_rows = total_pairs * len(epsilons)
    print(f"Total (n, trial) pairs: {total_pairs}")
    print(f"Epsilons per pair: {len(epsilons)}  ->  {total_rows} rows")
    print(f"Workers: {num_workers}")
    print(f"Sizes: {n_values}")
    print(f"Epsilons: {epsilons}\n")

    completed_per_n = {n: 0 for n in n_values}
    overall_start = time.perf_counter()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n", "trial", "seed", "epsilon",
            "exact_sum", "approx_sum", "ratio",
            "bound_lower", "bound_satisfied", "feasible",
        ])

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(run_trial, n, trial, seed, epsilons, max_weight): (n, trial)
                for (n, trial, seed) in tasks
            }

            for future in as_completed(futures):
                rows = future.result()
                for r in rows:
                    writer.writerow([
                        r["n"], r["trial"], r["seed"], r["epsilon"],
                        r["exact_sum"], r["approx_sum"], r["ratio"],
                        r["bound_lower"], r["bound_satisfied"], r["feasible"],
                    ])
                f.flush()

                n_done = rows[0]["n"]
                completed_per_n[n_done] += 1
                if completed_per_n[n_done] == trials_per_n:
                    elapsed = time.perf_counter() - overall_start
                    print(f"  n={n_done:>3} done ({trials_per_n} trials)  |  total elapsed: {elapsed:.1f}s")

    total = time.perf_counter() - overall_start
    print(f"\nAll done in {total:.1f}s ({total/60:.1f} min)")
    print(f"Wrote results to {output_path.resolve()}")


if __name__ == "__main__":
    N_VALUES = [10, 15, 20, 25, 30]
    EPSILONS = [0.5, 0.1, 0.01, 0.001]

    run_quality_experiment(
        n_values=N_VALUES,
        epsilons=EPSILONS,
        trials_per_n=30,
        max_weight=1000,
        output_csv="data/quality_data.csv",
        num_workers=8,
    )