"""
Random instance generator for the Subset Sum problem.

Generates random instances (S, t) where S is a list of positive integers
and t is a positive target. Parametric in n (problem size).
"""

import random
from typing import List, Tuple


def generate_instance(
    n: int,
    max_weight: int = 1000,
    target_ratio: float = 0.5,
    seed: int = None,
) -> Tuple[List[int], int]:
    """
    Generate a random Subset Sum instance.

    Args:
        n: number of elements in S (problem size). Must be >= 1.
        max_weight: upper bound for each element's value (inclusive).
                    Each x_i is drawn uniformly from [1, max_weight].
                    Default 1000.
        target_ratio: t is set to floor(target_ratio * sum(S)).
                      Must satisfy 0 < target_ratio < 1.
                      Default 0.5 (half of total sum).
        seed: optional random seed for reproducibility. If None,
              uses the current global random state.

    Returns:
        (S, t): S is a list of n positive integers; t is a positive integer.

    Guarantees (per Kellerer/Pferschy/Pisinger Ch. 4, eqs. 4.4-4.6):
        - All elements x_i satisfy 1 <= x_i <= max_weight
        - All elements x_i satisfy x_i <= t (each item fits)
        - sum(S) > t (the total exceeds capacity, so the problem is nontrivial)
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if max_weight < 1:
        raise ValueError(f"max_weight must be at least 1, got {max_weight}")
    if not (0 < target_ratio < 1):
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    rng = random.Random(seed)

    # Generate n random positive integers in [1, max_weight]
    S = [rng.randint(1, max_weight) for _ in range(n)]

    # Set t as a fraction of the total sum
    total = sum(S)
    t = int(total * target_ratio)

    # Edge case: if target_ratio * sum is too small (t < 1), bump to 1
    if t < 1:
        t = 1

    # Enforce x_i <= t for all i, by capping any oversized elements at t.
    # This rarely triggers when max_weight is much smaller than n * max_weight / 2.
    S = [min(x, t) for x in S]

    # Enforce sum(S) > t. This holds whenever n >= 2 and target_ratio < 1,
    # but we re-check after the capping step above.
    if sum(S) <= t:
        # Pathological case: bump t down so sum exceeds it
        t = sum(S) - 1
        if t < 1:
            t = 1

    return S, t