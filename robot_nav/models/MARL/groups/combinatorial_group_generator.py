"""
Combinatorial Group Generator — All Possible Subsets for n Robots.

Unlike ``group_generator.py`` (which uses binary allocation rules and produces
a structured but *incomplete* set of groups), this module generates **every**
possible combination of a given size from n robots.

For 14 robots with sizes 2 and 3:
    C(14, 2) =  91  pairs
    C(14, 3) = 364  triples
    Total    = 455  groups

Every robot appears in exactly the same number of groups, which keeps the
training data balanced — each robot gets equal representation as a group
member across the catalogue.

Usage
-----
    from robot_nav.models.MARL.groups.combinatorial_group_generator import (
        generate_all_combinations,
        print_combination_statistics,
    )
    groups = generate_all_combinations(n=14, min_size=2, max_size=3)
"""

from __future__ import annotations

from itertools import combinations
from typing import List


def generate_all_combinations(
    n: int,
    min_size: int = 2,
    max_size: int = 3,
) -> List[List[int]]:
    """
    Generate every possible robot group of the requested sizes.

    Args:
        n: Total number of robots (robot indices 0 … n-1).
        min_size: Smallest group size to include (inclusive).
        max_size: Largest group size to include (inclusive).

    Returns:
        Sorted list of groups, each group a sorted list of robot indices.
        Ordered by (size, lexicographic robot indices).

    Example:
        >>> groups = generate_all_combinations(n=6, min_size=2, max_size=3)
        >>> len(groups)   # C(6,2) + C(6,3) = 15 + 20 = 35
        35
    """
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if max_size > n:
        raise ValueError(
            f"max_size ({max_size}) cannot exceed n ({n})"
        )
    if min_size > max_size:
        raise ValueError(
            f"min_size ({min_size}) cannot exceed max_size ({max_size})"
        )

    groups: List[List[int]] = []
    for k in range(min_size, max_size + 1):
        for combo in combinations(range(n), k):
            groups.append(list(combo))

    return groups


def robot_frequency(
    groups: List[List[int]],
    n: int,
) -> List[int]:
    """
    Count how many groups each robot appears in.

    Args:
        groups: List of groups (each a list of robot indices).
        n: Total number of robots.

    Returns:
        List of length *n* where entry *i* is the number of groups
        containing robot *i*.
    """
    freq = [0] * n
    for g in groups:
        for idx in g:
            freq[idx] += 1
    return freq


def print_combination_statistics(
    groups: List[List[int]],
    n: int,
    max_display_per_size: int | None = 10,
) -> None:
    """
    Print a summary of the generated combinatorial groups.

    Args:
        groups: List of groups.
        n: Total number of robots.
        max_display_per_size: Cap on how many groups to print per size
            (``None`` = print all).
    """
    print(f"\n{'=' * 60}")
    print(f"Combinatorial Group Statistics ({n} robots)")
    print(f"{'=' * 60}")
    print(f"Total groups: {len(groups)}")

    # ---- Size distribution ----
    size_counts: dict[int, int] = {}
    for g in groups:
        s = len(g)
        size_counts[s] = size_counts.get(s, 0) + 1

    print("\nSize distribution:")
    for size in sorted(size_counts):
        count = size_counts[size]
        pct = count / len(groups) * 100
        print(f"  Size-{size}: {count:4d} groups ({pct:5.1f}%)")

    # ---- Per-robot frequency ----
    freq = robot_frequency(groups, n)
    print(f"\nPer-robot frequency (should be uniform):")
    print(f"  min={min(freq)}  max={max(freq)}  mean={sum(freq)/n:.1f}")

    # ---- Show sample groups ----
    print("\nSample groups by size:")
    for size in sorted(size_counts):
        of_size = [g for g in groups if len(g) == size]
        cap = max_display_per_size if max_display_per_size is not None else len(of_size)
        shown = of_size[:cap]
        remaining = len(of_size) - len(shown)
        print(f"  Size-{size} ({len(of_size)} total):")
        for g in shown:
            print(f"    {g}")
        if remaining > 0:
            print(f"    … ({remaining} more)")

    print(f"{'=' * 60}\n")


# -----------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 6 robots, size 2–3 ===")
    g6 = generate_all_combinations(n=6, min_size=2, max_size=3)
    print_combination_statistics(g6, n=6, max_display_per_size=5)

    print("=== 14 robots, size 2–3 ===")
    g14 = generate_all_combinations(n=14, min_size=2, max_size=3)
    print_combination_statistics(g14, n=14, max_display_per_size=5)
