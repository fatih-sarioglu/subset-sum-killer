import random
import time
from typing import List, Tuple

def trim(L: List[int], delta: float) -> List[int]:
    """
    Trim a sorted list L by parameter delta (0 < delta < 1).
    Keeps element y only if y > last_kept * (1 + delta).
    
    Reference: CLRS, page 1130.
    """
    if not L:
        return []
    
    L_trimmed = [L[0]]
    last = L[0]
    for i in range(1, len(L)):
        # L is sorted, so L[i] >= last
        if L[i] > last * (1 + delta):
            L_trimmed.append(L[i])
            last = L[i]
    return L_trimmed



def merge_lists(L1: List[int], L2: List[int]) -> List[int]:
    """
    Merge two sorted lists into one sorted list, removing duplicates.
    Runs in O(|L1| + |L2|) time.
    
    Reference: CLRS, page 1129 (MERGE-LISTS).
    """
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
    """
    FPTAS for Subset Sum optimization.
    Returns z such that z*/(1+epsilon) <= z <= z*, where z* is the true optimum.
    
    Reference: CLRS, page 1131 (APPROX-SUBSET-SUM).
    
    Args:
        S: list of positive integers
        t: target value
        epsilon: approximation parameter, 0 < epsilon < 1
    
    Returns:
        Approximate maximum subset sum (an integer).
    """
    n = len(S)
    L = [0]  # L_0 contains only the empty-subset sum
    delta = epsilon / (2 * n)
    
    for i in range(n):
        # Merge L with (L + x_i)
        L_shifted = [x + S[i] for x in L]
        L = merge_lists(L, L_shifted)
        # Trim
        L = trim(L, delta)
        # Remove elements > t
        L = [x for x in L if x <= t]
    
    return max(L)