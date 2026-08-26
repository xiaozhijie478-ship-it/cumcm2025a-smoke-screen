"""CUMCM 2025 A/Q3: optimize three bombs under strict joint full-surface coverage."""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

import q1_strict_occlusion as geometry
import q2_optimize as q2
import q4_optimize as q4


G = 9.8
FREE_FALL_LIMIT = math.sqrt(2 * q2.UAV_0[2] / G)
TIME_LIMIT = geometry.MISSILE_HIT_TIME

# Certified feasible incumbent found by the full-domain search followed by the
# four-dimensional relay refinement.  It is a warm start only; the search
# domain below remains the complete physical box.
INCUMBENT = [
    q2.Strategy(math.radians(179.84702991159443), 140.0, 0.0, 0.0, G),
    q2.Strategy(math.radians(179.84702991159443), 140.0, 7.403159406341729, 4.941303809307703, G),
    q2.Strategy(math.radians(179.84702991159443), 140.0, 10.868067802957235, 5.858496340286937, G),
]


def decode(x: np.ndarray) -> list[q2.Strategy]:
    """Map the unit box onto the complete physically feasible time domain."""
    theta, speed = map(float, x[:2])
    # Powers expand the narrow early-time region without excluding late times.
    release_1 = (TIME_LIMIT - 2.0) * float(x[2]) ** 4
    release_2 = release_1 + 1.0 + (TIME_LIMIT - release_1 - 2.0) * float(x[3]) ** 3
    release_3 = release_2 + 1.0 + (TIME_LIMIT - release_2 - 1.0) * float(x[4]) ** 3
    releases = [release_1, release_2, release_3]
    delays = [
        min(FREE_FALL_LIMIT, TIME_LIMIT - release) * float(x[5 + i]) ** 2
        for i, release in enumerate(releases)
    ]
    return [
        q2.Strategy(theta, speed, release + delay, delay, G)
        for release, delay in zip(releases, delays)
    ]


def encode(strategies: list[q2.Strategy]) -> np.ndarray:
    """Inverse of decode for feasible warm starts."""
    releases = [strategy.release_time for strategy in strategies]
    delays = [strategy.delay for strategy in strategies]
    x = np.zeros(8)
    x[0], x[1] = strategies[0].theta, strategies[0].speed
    x[2] = (releases[0] / (TIME_LIMIT - 2.0)) ** 0.25
    x[3] = ((releases[1] - releases[0] - 1.0) / (TIME_LIMIT - releases[0] - 2.0)) ** (1 / 3)
    x[4] = ((releases[2] - releases[1] - 1.0) / (TIME_LIMIT - releases[1] - 1.0)) ** (1 / 3)
    for i, (release, delay) in enumerate(zip(releases, delays)):
        x[5 + i] = math.sqrt(delay / min(FREE_FALL_LIMIT, TIME_LIMIT - release))
    return np.clip(x, [0, 70, 0, 0, 0, 0, 0, 0], [2 * math.pi, 140, 1, 1, 1, 1, 1, 1])


def sampled_intervals(
    strategy: q2.Strategy, dt: float, n_ring: int
) -> tuple[list[tuple[float, float]], float]:
    start, end = strategy.active_interval
    if end <= start:
        return [], math.inf
    count = max(2, int(math.ceil((end - start) / dt)) + 1)
    times = np.linspace(start, end, count)
    excess = q2.sampled_worst_distances(strategy, times, n_ring) - geometry.SMOKE_RADIUS
    pieces: list[tuple[float, float]] = []
    for t0, t1, f0, f1 in zip(times[:-1], times[1:], excess[:-1], excess[1:]):
        if f0 <= 0 and f1 <= 0:
            pieces.append((float(t0), float(t1)))
        elif f0 <= 0 < f1:
            root = t0 + (t1 - t0) * (-f0 / (f1 - f0))
            pieces.append((float(t0), float(root)))
        elif f1 <= 0 < f0:
            root = t0 + (t1 - t0) * (f0 / (f0 - f1))
            pieces.append((float(root), float(t1)))
    return geometry.merge(pieces), float(np.min(excess))


def union_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in geometry.merge(intervals))


def quick_components(
    x: np.ndarray, dt: float, n_ring: int
) -> tuple[float, float, list[float]]:
    intervals: list[tuple[float, float]] = []
    individual = 0.0
    minima: list[float] = []
    for strategy in decode(x):
        current, minimum = sampled_intervals(strategy, dt, n_ring)
        intervals.extend(current)
        individual += union_duration(current)
        minima.append(minimum)
    return union_duration(intervals), individual, minima


def guided_score(x: np.ndarray, dt: float, n_ring: int) -> float:
    """Union objective plus a small guide through the zero-coverage plateaus."""
    union, individual, minima = quick_components(x, dt, n_ring)
    near = sum(math.exp(-max(0.0, value) / 20.0) for value in minima)
    return -(union + 0.20 * individual + 0.50 * near)


def pure_score(x: np.ndarray, dt: float, n_ring: int) -> float:
    return -quick_components(x, dt, n_ring)[0]


def strict_score(x: np.ndarray, step: float) -> float:
    intervals = [pair for strategy in decode(x) for pair in q2.c_intervals(strategy, step)]
    return -union_duration(intervals)


def joint_components(
    x: np.ndarray, dt: float, points: np.ndarray
) -> tuple[float, list[tuple[float, float]], float]:
    strategies = decode(x)
    times = q4.time_grid(strategies, dt)
    excess = (
        q4.joint_worst_distances_chunked(strategies, times, points)
        - geometry.SMOKE_RADIUS
    )
    intervals = q4.sampled_intervals(times, excess)
    total = union_duration(intervals)
    near_values = np.zeros_like(times)
    outside = np.isfinite(excess) & (excess > 0.0)
    near_values[outside] = np.exp(-excess[outside] / 2.5)
    return total, intervals, float(np.trapezoid(near_values, times))


def joint_score(
    x: np.ndarray, dt: float, points: np.ndarray, guided: bool
) -> float:
    total, _, near = joint_components(x, dt, points)
    return -(total + (0.01 * near if guided else 0.0))


def joint_initial_population(seed: int, warm: np.ndarray, size: int) -> np.ndarray:
    bounds = [(0.0, 2 * math.pi), q2.SPEED_BOUNDS] + [(0.0, 1.0)] * 6
    rng = np.random.default_rng(seed)
    lower, upper = np.array([a for a, _ in bounds]), np.array([b for _, b in bounds])
    population = rng.uniform(lower, upper, size=(size, 8))
    local = size // 2
    scales = np.array(
        [math.radians(1.0), 3.0, 0.025, 0.040, 0.040, 0.045, 0.045, 0.045]
    )
    population[:local] = np.clip(
        warm + rng.normal(size=(local, 8)) * scales,
        lower,
        upper,
    )
    population[0] = warm
    return population


def optimize_joint(
    seeds: list[int], quick: bool
) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    bounds = [(0.0, 2 * math.pi), q2.SPEED_BOUNDS] + [(0.0, 1.0)] * 6
    warm = encode(INCUMBENT)
    coarse = q4.surface_points(16 if quick else 24, 4 if quick else 7, 3 if quick else 4)
    candidates = [warm]
    runs: list[tuple[int, float, float]] = []
    for seed in seeds:
        result = differential_evolution(
            lambda x: joint_score(x, 0.20 if quick else 0.12, coarse, True),
            bounds,
            seed=seed,
            init=joint_initial_population(seed, warm, 48 if quick else 96),
            maxiter=35 if quick else 80,
            tol=4e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        audit = q4.surface_points(40, 9, 6)
        total = joint_components(result.x, 0.04, audit)[0]
        candidates.append(result.x.copy())
        runs.append((seed, -float(result.fun), total))

    audit = q4.surface_points(48 if quick else 64, 11 if quick else 15, 8 if quick else 10)
    best = min(candidates, key=lambda x: joint_score(x, 0.025, audit, False))
    refined = minimize(
        lambda x: joint_score(x, 0.02 if quick else 0.012, audit, False),
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 280 if quick else 700, "xatol": 2e-8, "fatol": 2e-8},
    ).x
    return min([warm, best, refined], key=lambda x: joint_score(x, 0.01, audit, False)), runs


def decode_joint_relay(y: np.ndarray) -> list[q2.Strategy] | None:
    theta, explosion_2, delay_2, explosion_3, delay_3 = map(float, y)
    release_2, release_3 = explosion_2 - delay_2, explosion_3 - delay_3
    if (
        release_2 < 1.0
        or release_3 - release_2 < 1.0
        or explosion_3 < explosion_2
        or max(delay_2, delay_3) > FREE_FALL_LIMIT
    ):
        return None
    strategies = [
        q2.Strategy(theta, q2.SPEED_BOUNDS[1], 0.0, 0.0, G),
        q2.Strategy(theta, q2.SPEED_BOUNDS[1], explosion_2, delay_2, G),
        q2.Strategy(theta, q2.SPEED_BOUNDS[1], explosion_3, delay_3, G),
    ]
    if any(strategy.explosion_point[2] < 0.0 for strategy in strategies):
        return None
    return strategies


def joint_relay_score(
    y: np.ndarray, dt: float, points: np.ndarray, guided: bool
) -> float:
    strategies = decode_joint_relay(y)
    if strategies is None:
        return 1e3
    times = q4.time_grid(strategies, dt)
    excess = (
        q4.joint_worst_distances_chunked(strategies, times, points)
        - geometry.SMOKE_RADIUS
    )
    intervals = q4.sampled_intervals(times, excess)
    total = union_duration(intervals)
    near_values = np.zeros_like(times)
    outside = np.isfinite(excess) & (excess > 0.0)
    near_values[outside] = np.exp(-excess[outside] / 2.5)
    near = float(np.trapezoid(near_values, times))
    return -(total + (0.01 * near if guided else 0.0))


def optimize_joint_relay(seeds: list[int], quick: bool) -> tuple[list[q2.Strategy], list[tuple[int, float, float]]]:
    warm = np.array(
        [
            INCUMBENT[0].theta,
            INCUMBENT[1].explosion_time,
            INCUMBENT[1].delay,
            INCUMBENT[2].explosion_time,
            INCUMBENT[2].delay,
        ]
    )
    bounds = [
        (math.radians(176.0), math.radians(184.0)),
        (5.5, 9.5),
        (2.5, 7.5),
        (9.0, 14.5),
        (3.0, 9.5),
    ]
    coarse = q4.surface_points(20 if quick else 28, 5 if quick else 7, 3 if quick else 4)
    audit = q4.surface_points(64, 15, 10)
    candidates = [warm]
    runs: list[tuple[int, float, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        population = rng.uniform(
            [a for a, _ in bounds], [b for _, b in bounds], size=(40 if quick else 72, 5)
        )
        local = len(population) // 2
        scales = np.array([math.radians(0.5), 0.35, 0.35, 0.45, 0.45])
        population[:local] = np.clip(
            warm + rng.normal(size=(local, 5)) * scales,
            [a for a, _ in bounds],
            [b for _, b in bounds],
        )
        population[0] = warm
        result = differential_evolution(
            lambda y: joint_relay_score(y, 0.12 if quick else 0.07, coarse, True),
            bounds,
            seed=seed,
            init=population,
            maxiter=45 if quick else 100,
            tol=2e-5,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidate = decode_joint_relay(result.x)
        if candidate is not None:
            total = q4.metrics(candidate, 0.02, audit)[1]
            candidates.append(result.x.copy())
            runs.append((seed, -float(result.fun), total))
    best = min(candidates, key=lambda y: joint_relay_score(y, 0.012, audit, False))
    strategy = decode_joint_relay(best)
    assert strategy is not None
    return strategy, runs


def relay_candidate(
    y: np.ndarray, step: float
) -> tuple[list[q2.Strategy], list[list[tuple[float, float]]]] | None:
    """Build the active-boundary branch seen in strong Q3 solutions.

    The UAV uses its upper speed bound, releases bomb 1 immediately, and
    detonates each later bomb when the preceding strict interval ends.  Thus
    the eight physical controls reduce to theta and three fuse delays.  The
    one-second release constraints are still checked explicitly.
    """
    theta, *delays = map(float, y)
    if not 0.0 <= theta <= 2 * math.pi:
        return None
    if any(delay < 0.0 or delay > FREE_FALL_LIMIT for delay in delays):
        return None

    strategies: list[q2.Strategy] = []
    interval_groups: list[list[tuple[float, float]]] = []
    previous_release = -1.0
    explosion_time = delays[0]  # release_1 = 0 is the active boundary.
    for index, delay in enumerate(delays):
        if index:
            explosion_time = max(b for a, b in interval_groups[-1])
        release_time = explosion_time - delay
        if release_time < -1e-10 or release_time - previous_release < 1.0 - 1e-10:
            return None
        strategy = q2.Strategy(theta, q2.SPEED_BOUNDS[1], explosion_time, delay, G)
        intervals = q2.c_intervals(strategy, step)
        if not intervals:
            return None
        strategies.append(strategy)
        interval_groups.append(intervals)
        previous_release = release_time
    return strategies, interval_groups


def relay_score(y: np.ndarray, step: float) -> float:
    candidate = relay_candidate(y, step)
    if candidate is None:
        return 100.0
    return -union_duration([pair for group in candidate[1] for pair in group])


def relay_refine(warm: list[q2.Strategy]) -> list[q2.Strategy]:
    """Polish a full-domain solution on the four-dimensional relay branch."""
    y0 = np.array([warm[0].theta, *(strategy.delay for strategy in warm)])
    simplex = np.vstack(
        [
            y0,
            y0 + [0.0007, 0.0, 0.0, 0.0],
            y0 + [0.0, 0.035, 0.0, 0.0],
            y0 + [0.0, 0.0, 0.035, 0.0],
            y0 + [0.0, 0.0, 0.0, 0.035],
        ]
    )
    result = minimize(
        lambda y: relay_score(y, 0.10),
        y0,
        method="Nelder-Mead",
        bounds=[(0.0, 2 * math.pi), *((0.0, FREE_FALL_LIMIT),) * 3],
        options={
            "maxiter": 280,
            "xatol": 5e-9,
            "fatol": 5e-10,
            "initial_simplex": simplex,
        },
    )
    candidate = relay_candidate(result.x, 0.01)
    return warm if candidate is None else candidate[0]


def immediate_refine(warm: list[q2.Strategy]) -> list[q2.Strategy]:
    """Polish the active branch where bomb 1 is released and detonated at t=0."""
    y0 = np.array([warm[0].theta, warm[1].delay, warm[2].delay])
    simplex = np.vstack(
        [
            y0,
            y0 + [0.0007, 0.0, 0.0],
            y0 + [0.0, 0.035, 0.0],
            y0 + [0.0, 0.0, 0.035],
        ]
    )
    result = minimize(
        lambda y: relay_score(np.r_[y[0], 0.0, y[1:]], 0.10),
        y0,
        method="Nelder-Mead",
        bounds=[(0.0, 2 * math.pi), *((0.0, FREE_FALL_LIMIT),) * 2],
        options={
            "maxiter": 300,
            "xatol": 3e-9,
            "fatol": 3e-10,
            "initial_simplex": simplex,
        },
    )
    candidate = relay_candidate(np.r_[result.x[0], 0.0, result.x[1:]], 0.01)
    return warm if candidate is None else candidate[0]


def initial_population(seed: int, bounds: list[tuple[float, float]], warm: np.ndarray) -> np.ndarray:
    """Mix full-domain exploration with a certified feasible warm basin."""
    rng = np.random.default_rng(seed)
    population = rng.uniform([a for a, _ in bounds], [b for _, b in bounds], size=(96, 8))
    scales = np.array([math.radians(0.8), 2.0, 0.025, 0.035, 0.035, 0.035, 0.035, 0.035])
    population[:32] = np.clip(
        warm + rng.normal(size=(32, 8)) * scales,
        [a for a, _ in bounds],
        [b for _, b in bounds],
    )
    population[0] = warm
    return population


def optimize(seeds: list[int]) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    bounds = [
        (0.0, 2 * math.pi),
        q2.SPEED_BOUNDS,
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]
    warm = encode(INCUMBENT)
    candidates = [warm]
    runs: list[tuple[int, float, float]] = []
    for seed in seeds:
        result = differential_evolution(
            lambda x: guided_score(x, 0.12, 24),
            bounds,
            seed=seed,
            init=initial_population(seed, bounds, warm),
            maxiter=100,
            tol=3e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        union, _, _ = quick_components(result.x, 0.04, 64)
        candidates.append(result.x.copy())
        runs.append((seed, -float(result.fun), union))

    # Guidance ends here.  Every remaining ranking and refinement uses only
    # the requested union objective.
    ranked = sorted(candidates, key=lambda x: pure_score(x, 0.04, 64))[:3]
    refined: list[np.ndarray] = []
    for x0 in ranked:
        result = minimize(
            lambda x: pure_score(x, 0.035, 72),
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 550, "xatol": 2e-7, "fatol": 2e-7},
        )
        refined.append(result.x.copy())

    best = min(refined + [warm], key=lambda x: pure_score(x, 0.02, 96))
    fine = minimize(
        lambda x: pure_score(x, 0.02, 96),
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 450, "xatol": 8e-8, "fatol": 8e-8},
    ).x

    exact_start = min(refined + [warm, fine], key=lambda x: strict_score(x, 0.02))
    exact = minimize(
        lambda x: strict_score(x, 0.05),
        exact_start,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 180, "xatol": 5e-8, "fatol": 5e-9},
    ).x
    # A clipped Nelder-Mead simplex can stop just inside an active bound.
    # Compare the legal speed boundary explicitly instead of assuming it did.
    speed_limit = exact.copy()
    speed_limit[1] = q2.SPEED_BOUNDS[1]
    full_best = min([exact_start, exact, speed_limit], key=lambda x: strict_score(x, 0.01))

    # The unrestricted eight-dimensional search guards against a wrong active-
    # set assumption.  Only after it has found the basin do we exploit the
    # observed relay structure for a better-conditioned four-dimensional polish.
    relay = encode(relay_refine(decode(full_best)))
    immediate = encode(immediate_refine(decode(full_best)))
    return min([full_best, relay, immediate], key=lambda x: strict_score(x, 0.01)), runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="31,97,263")
    parser.add_argument("--incumbent-only", action="store_true")
    parser.add_argument("--relay-only", action="store_true")
    parser.add_argument("--immediate-only", action="store_true")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--joint-relay", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.joint_relay:
        strategies, runs = optimize_joint_relay(
            [int(item) for item in args.seeds.split(",")], args.quick
        )
        x = encode(strategies)
    elif args.joint:
        x, runs = optimize_joint([int(item) for item in args.seeds.split(",")], args.quick)
    elif args.incumbent_only:
        x, runs = encode(INCUMBENT), []
    elif args.relay_only:
        x, runs = encode(relay_refine(INCUMBENT)), []
    elif args.immediate_only:
        x, runs = encode(immediate_refine(INCUMBENT)), []
    else:
        x, runs = optimize([int(item) for item in args.seeds.split(",")])
    strategies = decode(x)
    strict = [q2.c_intervals(strategy, 0.01) for strategy in strategies]
    merged = geometry.merge([pair for intervals in strict for pair in intervals])

    print(f"coarse_runs={runs}")
    print(f"theta_deg={math.degrees(strategies[0].theta):.12f}")
    print(f"speed={strategies[0].speed:.12f}")
    for index, (strategy, intervals) in enumerate(zip(strategies, strict), 1):
        print(f"bomb_{index}")
        print(f"  release_time={strategy.release_time:.12f}")
        print(f"  release_point={strategy.release_point.tolist()}")
        print(f"  delay={strategy.delay:.12f}")
        print(f"  explosion_time={strategy.explosion_time:.12f}")
        print(f"  explosion_point={strategy.explosion_point.tolist()}")
        print(f"  strict_intervals={intervals}")
        print(f"  strict_duration={union_duration(intervals):.12f}")
    print(f"strict_union={merged}")
    print(f"strict_union_duration={union_duration(merged):.12f}")
    joint = q4.metrics(strategies, 0.005, q4.surface_points(80, 19, 12))
    print(f"joint_longest={joint[0]:.12f}")
    print(f"joint_total={joint[1]:.12f}")
    print(f"joint_intervals={joint[2]}")

    releases = [strategy.release_time for strategy in strategies]
    assert releases[1] - releases[0] >= 1 - 1e-9
    assert releases[2] - releases[1] >= 1 - 1e-9
    assert all(0 <= strategy.release_time <= strategy.explosion_time <= TIME_LIMIT for strategy in strategies)
    assert all(strategy.explosion_point[2] >= -1e-8 for strategy in strategies)
    assert union_duration(merged) <= sum(union_duration(item) for item in strict) + 1e-9
    assert joint[1] + 5e-4 >= union_duration(merged)


if __name__ == "__main__":
    main()
