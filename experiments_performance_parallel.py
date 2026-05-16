"""
Parallel performance testing for APPROX-SUBSET-SUM.

Distributes trials across multiple worker processes. Each trial generates
its own instance and times the algorithm independently. The CSV output
format matches the serial version, so downstream analysis is unchanged.
"""

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from sample_generator import generate_instance
from heuristic import approx_subset_sum


def run_single_trial(n: int, trial: int, seed: int, epsilon: float, max_weight: int) -> dict:
    """
    Run one trial: generate an instance, time the algorithm, return the result.
    Designed to be called inside a worker process.
    """
    S, t = generate_instance(n=n, max_weight=max_weight, seed=seed)
    start = time.perf_counter()
    result = approx_subset_sum(S, t, epsilon)
    elapsed = time.perf_counter() - start
    return {
        "n": n,
        "trial": trial,
        "seed": seed,
        "epsilon": epsilon,
        "runtime_seconds": elapsed,
        "result": result,
    }


def run_performance_experiment_parallel(
    n_values: List[int],
    trials_per_n: int = 30,
    epsilon: float = 0.1,
    max_weight: int = 1000,
    output_csv: str = "data/performance_data.csv",
    seed_base: int = 0,
    num_workers: int = 8,
) -> None:
    """
    Parallel version of the performance experiment.

    Uses `num_workers` worker processes. To keep timing measurements clean,
    we recommend num_workers ≤ (physical core count). Logical core count
    (os.cpu_count()) often double-counts SMT/hyperthreads.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the full list of tasks
    tasks: List[Tuple[int, int, int]] = []
    for n in n_values:
        for trial in range(trials_per_n):
            seed = seed_base + n * 10_000 + trial
            tasks.append((n, trial, seed))

    print(f"Total trials: {len(tasks)}")
    print(f"Workers: {num_workers}")
    print(f"Sizes: {n_values}")

    completed_per_n = {n: 0 for n in n_values}
    overall_start = time.perf_counter()

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "trial", "seed", "epsilon", "runtime_seconds", "result"])

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Submit all tasks
            futures = {
                pool.submit(run_single_trial, n, trial, seed, epsilon, max_weight): (n, trial)
                for (n, trial, seed) in tasks
            }

            # Collect results as they complete
            for future in as_completed(futures):
                row = future.result()
                writer.writerow([
                    row["n"], row["trial"], row["seed"], row["epsilon"],
                    row["runtime_seconds"], row["result"],
                ])
                f.flush()

                completed_per_n[row["n"]] += 1
                # Print a progress line whenever a size completes
                if completed_per_n[row["n"]] == trials_per_n:
                    elapsed = time.perf_counter() - overall_start
                    print(f"  n={row['n']:>5} done ({trials_per_n} trials)  |  total elapsed: {elapsed:.1f}s")

    total = time.perf_counter() - overall_start
    print(f"\nAll done in {total:.1f}s ({total/60:.1f} min)")
    print(f"Wrote results to {output_path.resolve()}")


if __name__ == "__main__":
    N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    run_performance_experiment_parallel(
        n_values=N_VALUES,
        trials_per_n=30,
        epsilon=0.1,
        max_weight=1000,
        output_csv="data/performance_data.csv",
        num_workers=8,
    )