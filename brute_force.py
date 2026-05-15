import random
import time
from typing import List, Tuple


def brute_force_subset_sum(S: List[int], t: int) -> Tuple[int, List[int]]:
    """
    Exact solution to the Subset Sum optimization problem via bitmask enumeration.
    
    Args:
        S: list of positive integers
        t: target value (positive integer)
    
    Returns:
        (best_sum, best_subset) where best_sum is the maximum sum ≤ t
        achievable by any subset of S, and best_subset is one such subset.
    """
    n = len(S)
    best_sum = 0
    best_subset = []
    
    # Iterate over all 2^n possible subsets
    for mask in range(1 << n):  # 1 << n == 2^n
        current_sum = 0
        current_subset = []
        for i in range(n):
            if mask & (1 << i):  # bit i is set => include S[i]
                current_sum += S[i]
                current_subset.append(S[i])
                if current_sum > t:
                    break  # prune: no need to continue this mask
        
        if current_sum <= t and current_sum > best_sum:
            best_sum = current_sum
            best_subset = current_subset
    
    return best_sum, best_subset