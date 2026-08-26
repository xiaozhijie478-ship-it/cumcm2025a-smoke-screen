"""Deterministic full-angle audit for Q3's immediate-explosion relay branch.

DIRECT partitions the complete heading domain and only has to discover a
feasible basin.  Observer C and Q3's boundary refiner remain authoritative.
This is a deterministic global-search audit, not a proof of global optimality.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import direct

import q2_optimize as q2
import q3_optimize as q3


FIRST = q2.Strategy(0.0, q2.SPEED_BOUNDS[1], 0.0, 0.0, q3.G)
FIRST_INTERVAL = q2.c_intervals(FIRST, 0.01)[0]


def candidate(x: np.ndarray) -> tuple[list[q2.Strategy], float] | None:
    """Decode (heading, second delay, third-delay fraction)."""
    theta, delay_2, fraction_3 = map(float, x)
    first_start, explosion_2 = FIRST_INTERVAL
    first = q2.Strategy(theta, q2.SPEED_BOUNDS[1], 0.0, 0.0, q3.G)
    release_2 = explosion_2 - delay_2
    second = q2.Strategy(theta, q2.SPEED_BOUNDS[1], explosion_2, delay_2, q3.G)
    violation_2, _, _ = q2.c_observer(second, explosion_2)
    if violation_2 > 0:
        return None
    intervals_2 = q2.c_intervals(second, 0.10)
    if not intervals_2:
        return None

    explosion_3 = max(b for _, b in intervals_2)
    max_delay_3 = min(q3.FREE_FALL_LIMIT, explosion_3 - release_2 - 1.0)
    if max_delay_3 < 0:
        return None
    delay_3 = fraction_3 * max_delay_3
    third = q2.Strategy(theta, q2.SPEED_BOUNDS[1], explosion_3, delay_3, q3.G)
    violation_3, _, _ = q2.c_observer(third, explosion_3)
    if violation_3 > 0:
        return None
    intervals_3 = q2.c_intervals(third, 0.10)
    if not intervals_3:
        return None
    return [first, second, third], max(b for _, b in intervals_3) - first_start


def score(x: np.ndarray) -> float:
    decoded = candidate(x)
    if decoded is not None:
        return -decoded[1]
    # A continuous guide makes the very narrow full-occlusion cone discoverable.
    theta, delay_2, _ = map(float, x)
    second = q2.Strategy(theta, q2.SPEED_BOUNDS[1], FIRST_INTERVAL[1], delay_2, q3.G)
    return 10.0 + q2.c_observer(second, FIRST_INTERVAL[1])[2]


def deterministic_candidate(maxfun: int) -> tuple[list[q2.Strategy], float, int]:
    result = direct(
        score,
        [(0.0, 2 * math.pi), (0.0, FIRST_INTERVAL[1] - 1.0), (0.0, 1.0)],
        maxfun=maxfun,
        maxiter=maxfun,
        locally_biased=False,
        len_tol=2e-5,
    )
    decoded = candidate(result.x)
    if decoded is None:
        raise RuntimeError("DIRECT did not discover a feasible relay basin")
    return decoded[0], decoded[1], int(result.nfev)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxfun", type=int, default=15_000)
    args = parser.parse_args()
    coarse, coarse_duration, evaluations = deterministic_candidate(args.maxfun)
    polished = q3.immediate_refine(coarse)
    strict = [q2.c_intervals(strategy, 0.002) for strategy in polished]
    duration = q3.union_duration([pair for group in strict for pair in group])

    print(f"direct_evaluations={evaluations}")
    print(f"direct_strict_duration={coarse_duration:.12f}")
    print(f"polished_strict_duration={duration:.12f}")
    print(f"theta_deg={math.degrees(polished[0].theta):.12f}")
    for index, (strategy, intervals) in enumerate(zip(polished, strict), 1):
        print(
            f"bomb_{index}: release={strategy.release_time:.12f}, "
            f"delay={strategy.delay:.12f}, explosion={strategy.explosion_time:.12f}, "
            f"intervals={intervals}"
        )

    releases = [strategy.release_time for strategy in polished]
    assert all(abs(strategy.theta - polished[0].theta) < 1e-12 for strategy in polished)
    assert releases[1] - releases[0] >= 1 - 1e-9
    assert releases[2] - releases[1] >= 1 - 1e-9
    assert duration > 7.64


if __name__ == "__main__":
    main()
