import random
import time
from typing import List, Tuple

def trim(L: List[int], delta: float) -> List[int]:
    """Trim a sorted list L by factor delta (0 < delta < 1)."""
    if not L:
        return []
    
    L_trimmed = [L[0]]
    last = L[0]
    for i in range(1, len(L)):
        # Keep values spaced by a (1 + delta) factor.
        if L[i] > last * (1 + delta):
            L_trimmed.append(L[i])
            last = L[i]
    return L_trimmed



def merge_lists(L1: List[int], L2: List[int]) -> List[int]:
    """Merge two sorted lists into one, removing duplicates."""
    merged = []
    i, j = 0, 0
    while i < len(L1) and j < len(L2):
        if L1[i] < L2[j]:
            if not merged or merged[-1] != L1[i]:
                merged.append(L1[i])
            i += 1
        elif L1[i] > L2[j]:
            if not merged or merged[-1] != L2[j]:
                merged.append(L2[j])
            j += 1
        else:  # equal - take once, advance both
            if not merged or merged[-1] != L1[i]:
                merged.append(L1[i])
            i += 1
            j += 1
    while i < len(L1):
        if not merged or merged[-1] != L1[i]:
            merged.append(L1[i])
        i += 1
    while j < len(L2):
        if not merged or merged[-1] != L2[j]:
            merged.append(L2[j])
        j += 1
    return merged



def approx_subset_sum(S: List[int], t: int, epsilon: float) -> int:
    """Approximate subset sum within the (1 + epsilon) factor."""
    n = len(S)
    L = [0]
    delta = epsilon / (2 * n)
    
    for i in range(n):
        L_shifted = [x + S[i] for x in L]
        L = merge_lists(L, L_shifted)
        L = trim(L, delta)
        L = [x for x in L if x <= t]
    
    return max(L)