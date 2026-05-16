"""
Performance testing for APPROX-SUBSET-SUM.

Measures runtime across a range of input sizes, with many trials per size,
and writes raw timings to CSV for downstream statistical analysis.
"""

import csv
import time
from pathlib import Path
from typing import List

from sample_generator import generate_instance
from heuristic import approx_subset_sum


def run_performance_experiment(
    n_values: List[int],
    trials_per_n: int = 30,
    epsilon: float = 0.1,
    max_weight: int = 1000,
    output_csv: str = "performance_data.csv",
    seed_base: int = 0,
) -> None:
    """
    Run timing trials of APPROX-SUBSET-SUM across multiple input sizes.

    For each n in n_values, runs `trials_per_n` independent trials. Each
    trial generates a fresh random instance (using a distinct seed) and
    times one call to approx_subset_sum.

    Writes one row per trial to output_csv with columns:
        n, trial, seed, runtime_seconds, result

    Args:
        n_values: list of problem sizes to test.
        trials_per_n: number of independent trials per size.
        epsilon: approximation parameter (fixed across the entire sweep).
        max_weight: passed to generate_instance.
        output_csv: path to write raw timing data.
        seed_base: starting offset for seeds (lets you re-run with fresh
            seeds if needed without overlapping previous runs).
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "trial", "seed", "epsilon", "runtime_seconds", "result"])

        for n in n_values:
            print(f"Running n={n}...", end=" ", flush=True)
            n_start = time.perf_counter()

            for trial in range(trials_per_n):
                seed = seed_base + n * 10_000 + trial
                S, t = generate_instance(n=n, max_weight=max_weight, seed=seed)

                start = time.perf_counter()
                result = approx_subset_sum(S, t, epsilon)
                elapsed = time.perf_counter() - start

                writer.writerow([n, trial, seed, epsilon, elapsed, result])
                f.flush()  # write incrementally so partial runs are preserved

            n_total = time.perf_counter() - n_start
            print(f"done ({n_total:.1f}s for {trials_per_n} trials)")

    print(f"\nWrote results to {output_path.resolve()}")


if __name__ == "__main__":
    # Default sweep - adjust as needed
    N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    run_performance_experiment(
        n_values=N_VALUES,
        trials_per_n=30,
        epsilon=0.1,
        max_weight=1000,
        output_csv="data/performance_data.csv",
    )