"""Random instance generator for Subset Sum."""

import random
from typing import List, Tuple


def generate_instance(
    n: int,
    max_weight: int = 1000,
    target_ratio: float = 0.5,
    seed: int = None,
) -> Tuple[List[int], int]:
    """Generate a random Subset Sum instance.

    Returns (S, t) where S has positive integers and t is a positive target.
    We cap any item that exceeds t and make sure sum(S) > t.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if max_weight < 1:
        raise ValueError(f"max_weight must be at least 1, got {max_weight}")
    if not (0 < target_ratio < 1):
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    rng = random.Random(seed)

    S = [rng.randint(1, max_weight) for _ in range(n)]

    total = sum(S)
    t = int(total * target_ratio)

    if t < 1:
        t = 1

    # Cap items so each one fits.
    S = [min(x, t) for x in S]

    if sum(S) <= t:
        t = sum(S) - 1
        if t < 1:
            t = 1

    return S, t