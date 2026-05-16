"""
Functional testing for the Subset Sum implementations.

Applies the testing methodology covered in the CS301 Test Design lecture
(Yenigün, slides 6–8):
  - Black-box equivalence-class partitioning and boundary-value analysis
  - White-box statement/branch coverage via targeted code-path tests
  - Cross-validation with brute force as oracle (random small instances)
"""

from dataclasses import dataclass
from typing import List, Tuple

from brute_force import brute_force_subset_sum
from heuristic import approx_subset_sum, trim, merge_lists
from sample_generator import generate_instance


# ----------------------------------------------------------------------------
# Phase 1: Hand-designed test cases
# ----------------------------------------------------------------------------

@dataclass
class TestCase:
    test_id: str
    category: str        # "BB-EC", "BB-BV", "WB-Branch", "BB-EC-eps"
    description: str
    S: List[int]
    t: int
    epsilon: float
    expected_exact: int  # the known true optimum z*


HAND_TESTS: List[TestCase] = [
    # --- Black-box: equivalence classes ---
    TestCase("T01", "BB-EC",      "Single element, fits",                 [7],              10, 0.1, 7),
    TestCase("T02", "BB-EC",      "Single element, exceeds t",            [15],             10, 0.1, 0),
    TestCase("T03", "BB-EC",      "Single element, exact fit",            [10],             10, 0.1, 10),
    TestCase("T04", "BB-EC",      "All duplicates",                       [5, 5, 5, 5],     12, 0.1, 10),
    TestCase("T05", "BB-EC",      "All elements equal to t",              [10, 10, 10],     10, 0.1, 10),
    TestCase("T06", "BB-EC",      "Pre-sorted ascending",                 [1, 2, 3, 4, 5],   8, 0.1, 8),
    TestCase("T07", "BB-EC",      "Pre-sorted descending",                [5, 4, 3, 2, 1],   8, 0.1, 8),
    TestCase("T08", "BB-EC",      "Random order (same multiset as T06)",  [3, 1, 5, 2, 4],   8, 0.1, 8),

    # --- Black-box: boundary values for t ---
    TestCase("T09", "BB-BV",      "t = min(S)",                           [3, 5, 7],         3, 0.1, 3),
    TestCase("T10", "BB-BV",      "t = min(S) - 1",                       [3, 5, 7],         2, 0.1, 0),
    TestCase("T11", "BB-BV",      "t = sum(S) - 1",                       [3, 5, 7],        14, 0.1, 12),
    TestCase("T12", "BB-BV",      "t = sum(S)",                           [3, 5, 7],        15, 0.1, 15),
    TestCase("T13", "BB-BV",      "t > sum(S)",                           [3, 5, 7],       100, 0.1, 15),

    # --- White-box: targeted branches ---
    TestCase("T14", "WB-Branch",  "BF prune branch (current_sum > t)",    [50, 50, 50, 50], 60, 0.1, 50),
    TestCase("T15", "WB-Branch",  "Heuristic x>t filter branch",          [1, 5, 100],      10, 0.5, 6),
    TestCase("T16", "WB-Branch",  "MERGE-LISTS equal-element branch",     [5, 5],           10, 0.1, 10),
    TestCase("T17", "WB-Branch",  "TRIM keep branch (large gaps)",        [1, 100, 1000],  999, 0.5, 101),
    TestCase("T18", "WB-Branch",  "TRIM discard branch (close values)",   [10, 11, 12, 13], 25, 0.5, 25),

    # --- Black-box: equivalence classes for epsilon ---
    TestCase("T19", "BB-EC-eps",  "epsilon = 0.5 (large)",                [10, 20, 30, 40], 50, 0.5,   50),
    TestCase("T20", "BB-EC-eps",  "epsilon = 0.1 (medium)",               [10, 20, 30, 40], 50, 0.1,   50),
    TestCase("T21", "BB-EC-eps",  "epsilon = 0.01 (small)",               [10, 20, 30, 40], 50, 0.01,  50),
    TestCase("T22", "BB-EC-eps",  "epsilon = 0.001 (very small)",         [10, 20, 30, 40], 50, 0.001, 50),
]


@dataclass
class TestResult:
    tc: TestCase
    actual_exact: int
    actual_approx: int
    exact_pass: bool
    approx_in_bound: bool
    approx_feasible: bool
    overall_pass: bool


def run_test_case(tc: TestCase) -> TestResult:
    actual_exact, _ = brute_force_subset_sum(tc.S, tc.t)
    actual_approx = approx_subset_sum(tc.S, tc.t, tc.epsilon)

    exact_pass = (actual_exact == tc.expected_exact)
    if actual_exact > 0:
        lower = actual_exact / (1.0 + tc.epsilon)
        approx_in_bound = (lower - 1e-9) <= actual_approx <= actual_exact
    else:
        approx_in_bound = (actual_approx == 0)
    approx_feasible = (actual_approx <= tc.t) and (actual_approx <= actual_exact)
    overall = exact_pass and approx_in_bound and approx_feasible

    return TestResult(
        tc=tc,
        actual_exact=actual_exact,
        actual_approx=actual_approx,
        exact_pass=exact_pass,
        approx_in_bound=approx_in_bound,
        approx_feasible=approx_feasible,
        overall_pass=overall,
    )


def print_results_table(results: List[TestResult]) -> None:
    header = (
        f"{'ID':<5} {'Category':<11} {'Description':<40} "
        f"{'exp':>5} {'BF':>5} {'AP':>5} "
        f"{'BF?':>4} {'AP_ok?':>7} {'feas?':>6} {'PASS':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        mark = "✓" if r.overall_pass else "✗"
        print(
            f"{r.tc.test_id:<5} {r.tc.category:<11} {r.tc.description:<40} "
            f"{r.tc.expected_exact:>5} {r.actual_exact:>5} {r.actual_approx:>5} "
            f"{('✓' if r.exact_pass else '✗'):>4} "
            f"{('✓' if r.approx_in_bound else '✗'):>7} "
            f"{('✓' if r.approx_feasible else '✗'):>6} "
            f"{mark:>5}"
        )
    passed = sum(1 for r in results if r.overall_pass)
    print()
    print(f"Hand-designed tests passed: {passed}/{len(results)}")


# ----------------------------------------------------------------------------
# Phase 2: Direct unit tests on helper functions
# ----------------------------------------------------------------------------

def run_helper_unit_tests() -> Tuple[int, int]:
    cases = [
        # --- trim ---
        ("trim empty",                    trim([], 0.1),                       []),
        ("trim singleton",                trim([5], 0.1),                      [5]),
        ("trim discards",                 trim([1, 1.05, 1.2, 1.5], 0.1),      [1, 1.2, 1.5]),
        ("trim keeps all",                trim([1, 10, 100, 1000], 0.1),       [1, 10, 100, 1000]),
        ("trim delta=0.5",                trim([1, 2, 3, 4, 5], 0.5),          [1, 2, 4]),
        # --- merge_lists: basic cases ---
        ("merge both empty",              merge_lists([], []),                 []),
        ("merge L1 empty",                merge_lists([], [1, 2, 3]),          [1, 2, 3]),
        ("merge L2 empty",                merge_lists([1, 2, 3], []),          [1, 2, 3]),
        ("merge disjoint",                merge_lists([1, 3, 5], [2, 4, 6]),   [1, 2, 3, 4, 5, 6]),
        ("merge with duplicates",         merge_lists([1, 2, 3], [2, 3, 4]),   [1, 2, 3, 4]),
        ("merge L1 longer",               merge_lists([1, 2, 3, 4, 5], [10]),  [1, 2, 3, 4, 5, 10]),
        # --- merge_lists: defensive duplicate-skip branches ---
        ("merge L1 internal duplicate",   merge_lists([1, 1, 2], [3]),         [1, 2, 3]),
        ("merge L2 internal duplicate",   merge_lists([3], [1, 1, 2]),         [1, 2, 3]),
        ("merge equal-branch duplicate",  merge_lists([1, 1], [1, 1]),         [1]),
        ("merge L1 tail with duplicate",  merge_lists([1, 1], []),             [1]),
        ("merge L2 tail with duplicate",  merge_lists([], [1, 1]),             [1]),
    ]
    failures = 0
    for name, got, want in cases:
        if got == want:
            print(f"  ✓ {name}: got {got}")
        else:
            print(f"  ✗ {name}: got {got}, want {want}")
            failures += 1
    return len(cases) - failures, len(cases)


# ----------------------------------------------------------------------------
# Phase 3: Random cross-validation (brute force as oracle)
# ----------------------------------------------------------------------------

def random_cross_validation(
    n_values: Tuple[int, ...] = (10, 12, 14, 16),
    trials_per_n: int = 20,
    epsilons: Tuple[float, ...] = (0.5, 0.1, 0.01, 0.001),
    max_weight: int = 1000,
    seed_base: int = 2_000_000,
) -> dict:
    """Verifies, on many small random instances, that the heuristic obeys
    its three runtime invariants. Uses a different seed_base from Sections 6
    and 7 to draw fresh instances."""
    total = 0
    failures = []
    for n in n_values:
        for trial in range(trials_per_n):
            seed = seed_base + n * 10_000 + trial
            S, t = generate_instance(n=n, max_weight=max_weight, seed=seed)
            exact, _ = brute_force_subset_sum(S, t)
            for eps in epsilons:
                approx = approx_subset_sum(S, t, eps)
                total += 1
                if approx > exact:
                    failures.append((n, trial, seed, eps, "approx > exact", exact, approx))
                if approx > t:
                    failures.append((n, trial, seed, eps, "approx > t",     exact, approx))
                if exact > 0 and approx < exact / (1 + eps) - 1e-9:
                    failures.append((n, trial, seed, eps, "bound violation", exact, approx))
    return {"total_trials": total, "failures": failures}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("FUNCTIONAL TESTING - SUBSET SUM")
    print("=" * 90)
    print()

    print("Phase 1: Hand-designed test cases (equivalence classes, boundary values, white-box branches)")
    print("-" * 90)
    results = [run_test_case(tc) for tc in HAND_TESTS]
    print_results_table(results)
    print()

    print("Phase 2: Helper-function unit tests (TRIM, MERGE-LISTS)")
    print("-" * 90)
    helper_pass, helper_total = run_helper_unit_tests()
    print()
    print(f"Helper unit tests passed: {helper_pass}/{helper_total}")
    print()

    print("Phase 3: Random cross-validation (brute force as oracle)")
    print("-" * 90)
    cv = random_cross_validation()
    print(f"Trials (n, trial, ε) checked: {cv['total_trials']}")
    print(f"Invariant violations: {len(cv['failures'])}")
    if cv["failures"]:
        for f in cv["failures"][:10]:
            print(f"  {f}")
        if len(cv["failures"]) > 10:
            print(f"  ... and {len(cv['failures']) - 10} more")
    else:
        print("✓ All invariants hold on every random instance")
    print()

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    hand_pass = sum(1 for r in results if r.overall_pass)
    print(f"  Hand-designed tests:  {hand_pass}/{len(results)} passed")
    print(f"  Helper unit tests:    {helper_pass}/{helper_total} passed")
    print(f"  Cross-validation:     {cv['total_trials'] - len(cv['failures'])}/{cv['total_trials']} clean")


if __name__ == "__main__":
    main()