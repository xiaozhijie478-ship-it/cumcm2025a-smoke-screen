"""Search pair synergy and audit three-ball coverage order for CUMCM 2025 A/Q4."""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
from scipy.optimize import differential_evolution, minimize

import q4_optimize as q4


def decode_anchor(x: np.ndarray, names: list[str]) -> tuple[float, list[q4.Strategy]]:
    """Decode an anchor time and three strategies that are active at that time."""
    t = float(x[0])
    strategies: list[q4.Strategy] = []
    for index, name in enumerate(names):
        theta, speed, age_fraction, delay_fraction = map(
            float, x[1 + 4 * index : 5 + 4 * index]
        )
        age = age_fraction * min(q4.geometry.SMOKE_LIFETIME, t)
        explosion_time = t - age
        free_fall = math.sqrt(2 * q4.UAVS[name][2] / q4.G)
        delay = delay_fraction * min(explosion_time, free_fall)
        strategies.append(
            q4.Strategy(
                name,
                q4.UAVS[name],
                theta % (2 * math.pi),
                speed,
                explosion_time,
                delay,
            )
        )
    return t, strategies


def responsibility(
    strategies: list[q4.Strategy], t: float, points: np.ndarray, temperature: float = 0.7
) -> dict[str, object]:
    """Return joint margin, individual margins, and surface responsibility shares."""
    times = np.array([t])
    distances = np.stack(
        [q4.joint_point_distances([strategy], times, points)[0] for strategy in strategies]
    )
    best = np.min(distances, axis=0)
    hard_owner = np.argmin(distances, axis=0)
    shifted = np.clip((distances - best) / temperature, 0.0, 80.0)
    soft = np.exp(-shifted)
    soft /= np.sum(soft, axis=0, keepdims=True)
    return {
        "joint_margin": float(np.max(best) - q4.geometry.SMOKE_RADIUS),
        "individual_margins": np.max(distances, axis=1) - q4.geometry.SMOKE_RADIUS,
        "individual_best_margins": np.min(distances, axis=1) - q4.geometry.SMOKE_RADIUS,
        "hard_shares": np.array([np.mean(hard_owner == i) for i in range(len(strategies))]),
        "soft_shares": np.mean(soft, axis=1),
        "uncovered": float(np.mean(best > q4.geometry.SMOKE_RADIUS)),
        "overlap": float(
            np.mean(np.sum(distances <= q4.geometry.SMOKE_RADIUS, axis=0) >= 2)
        ),
        "best_distances": best,
    }


def synergy_loss(x: np.ndarray, points: np.ndarray, names: list[str]) -> float:
    t, strategies = decode_anchor(x, names)
    stats = responsibility(strategies, t, points)
    excess = np.maximum(
        np.asarray(stats["best_distances"]) - q4.geometry.SMOKE_RADIUS,
        0.0,
    )
    shares = np.asarray(stats["soft_shares"])
    individual = np.asarray(stats["individual_margins"])
    individual_best = np.asarray(stats["individual_best_margins"])

    coverage = float(np.mean(excess**2) + 0.35 * np.max(excess) ** 2)
    # Require every ball to own a visible part of the surface and prevent the
    # trivial solution in which one ball covers the whole cylinder alone.
    participation_penalty = 0.6 * float(
        np.sum(np.maximum(individual_best + 0.50, 0.0) ** 2)
    )
    share_penalty = 600.0 * float(np.sum(np.maximum(0.08 - shares, 0.0) ** 2))
    necessity_penalty = 0.8 * float(np.sum(np.maximum(0.20 - individual, 0.0) ** 2))
    return coverage + participation_penalty + share_penalty + necessity_penalty


def bounds(names: list[str], time_range: tuple[float, float]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = [time_range]
    for _ in names:
        result.extend([(0.0, 2 * math.pi), q4.q2.SPEED_BOUNDS, (0.0, 1.0), (0.0, 1.0)])
    return result


def search(
    seed: int,
    quick: bool,
    names: list[str],
    time_range: tuple[float, float],
) -> tuple[float, list[q4.Strategy], float]:
    coarse = q4.surface_points(20 if quick else 28, 5 if quick else 7, 3 if quick else 4)
    result = differential_evolution(
        lambda x: synergy_loss(x, coarse, names),
        bounds(names, time_range),
        seed=seed,
        popsize=8 if quick else 12,
        maxiter=65 if quick else 130,
        tol=2e-5,
        polish=False,
        workers=1,
        updating="immediate",
    )
    medium = q4.surface_points(48 if quick else 64, 11 if quick else 15, 8 if quick else 10)
    refined = minimize(
        lambda x: synergy_loss(x, medium, names),
        result.x,
        method="Nelder-Mead",
        bounds=bounds(names, time_range),
        options={"maxiter": 500 if quick else 1000, "xatol": 2e-8, "fatol": 2e-9},
    )
    t, strategies = decode_anchor(refined.x, names)
    return t, strategies, float(refined.fun)


def decode_subset(x: np.ndarray, names: list[str]) -> list[q4.Strategy]:
    return [q4.decode_one(name, x[4 * i : 4 * i + 4]) for i, name in enumerate(names)]


def pair_total_stats(
    strategies: list[q4.Strategy], dt: float, points: np.ndarray
) -> tuple[float, float, float, list[tuple[float, float]], float]:
    times = q4.time_grid(strategies, dt)
    joint_excess = (
        q4.joint_worst_distances_chunked(strategies, times, points)
        - q4.geometry.SMOKE_RADIUS
    )
    joint_intervals = q4.sampled_intervals(times, joint_excess)
    joint_total = sum(b - a for a, b in joint_intervals)

    individual_pieces: list[tuple[float, float]] = []
    for strategy in strategies:
        excess = (
            q4.joint_worst_distances_chunked([strategy], times, points)
            - q4.geometry.SMOKE_RADIUS
        )
        individual_pieces.extend(q4.sampled_intervals(times, excess))
    individual_total = sum(b - a for a, b in q4.geometry.merge(individual_pieces))
    synergy = max(0.0, joint_total - individual_total)

    near_values = np.zeros_like(times)
    finite = np.isfinite(joint_excess)
    outside = finite & (joint_excess > 0.0)
    near_values[outside] = np.exp(-joint_excess[outside] / 2.5)
    near = float(np.trapezoid(near_values, times))
    return joint_total, individual_total, synergy, joint_intervals, near


def interval_difference(
    minuend: list[tuple[float, float]],
    subtrahend: list[tuple[float, float]],
    tol: float = 1e-6,
) -> list[tuple[float, float]]:
    """Return the parts of ``minuend`` not covered by ``subtrahend``."""
    result: list[tuple[float, float]] = []
    removed = q4.geometry.merge(subtrahend)
    for start, end in q4.geometry.merge(minuend):
        cursor = start
        for left, right in removed:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left - cursor > tol:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if end - cursor > tol:
            result.append((cursor, end))
    return result


def coverage_order_stats(
    strategies: list[q4.Strategy], dt: float, points: np.ndarray
) -> dict[str, object]:
    """Split coverage into single-, pair-, and triple-essential time sets."""
    if len(strategies) != 3:
        raise ValueError("coverage-order audit requires exactly three strategies")
    times = q4.time_grid(strategies, dt)

    unions: dict[int, list[tuple[float, float]]] = {}
    for order in (1, 2, 3):
        pieces: list[tuple[float, float]] = []
        for subset in itertools.combinations(strategies, order):
            excess = (
                q4.joint_worst_distances_chunked(list(subset), times, points)
                - q4.geometry.SMOKE_RADIUS
            )
            pieces.extend(q4.sampled_intervals(times, excess))
        unions[order] = q4.geometry.merge(pieces)

    single = unions[1]
    up_to_pair = q4.geometry.merge(unions[1] + unions[2])
    triple = unions[3]
    pair_only = interval_difference(up_to_pair, single)
    triple_only = interval_difference(triple, up_to_pair)
    duration = lambda intervals: sum(end - start for start, end in intervals)
    return {
        "single_intervals": single,
        "pair_only_intervals": pair_only,
        "triple_only_intervals": triple_only,
        "joint_intervals": triple,
        "single_duration": duration(single),
        "pair_only_duration": duration(pair_only),
        "triple_only_duration": duration(triple_only),
        "joint_duration": duration(triple),
    }


def triple_reachability_certificate() -> dict[str, float | bool]:
    """Certify that FY1 and FY3 cannot contribute at the same instant.

    A cloud can help only if its radius-10 ball reaches at least one finite
    missile-to-target segment.  Horizontal coordinate bounds alone suffice:
    FY1 cannot move left fast enough after the missile passes it, while FY3
    cannot reach the segment's nonnegative-y half-plane early enough.
    """
    vmax = float(q4.q2.SPEED_BOUNDS[1])
    missile_vx = float(q4.geometry.MISSILE_SPEED * q4.geometry.MISSILE_DIRECTION[0])
    fy1_latest = float(
        (q4.UAVS["FY1"][0] - q4.geometry.MISSILE_0[0] - q4.geometry.SMOKE_RADIUS)
        / (vmax + missile_vx)
    )
    target_min_y = float(q4.geometry.TARGET_CENTER_XY[1] - q4.geometry.TARGET_RADIUS)
    segment_min_y = min(float(q4.geometry.MISSILE_0[1]), target_min_y)
    fy3_earliest = float(
        (segment_min_y - q4.geometry.SMOKE_RADIUS - q4.UAVS["FY3"][1]) / vmax
    )
    return {
        "fy1_latest_possible_contribution": fy1_latest,
        "fy3_earliest_possible_contribution": fy3_earliest,
        "gap": fy3_earliest - fy1_latest,
        "triple_exclusive_possible": fy1_latest >= fy3_earliest,
    }


def known_synergy_candidate(names: list[str]) -> list[q4.Strategy]:
    if names == ["FY1", "FY2"]:
        return [
            q4.Strategy("FY1", q4.UAVS["FY1"], math.radians(179.521742949463), 139.992710707090, 10.369094794536, 5.666652214316),
            q4.Strategy("FY2", q4.UAVS["FY2"], math.radians(278.840162145426), 132.559294026831, 10.323682648348, 5.878151957756),
        ]
    if names == ["FY2", "FY3"]:
        return [
            q4.Strategy("FY2", q4.UAVS["FY2"], math.radians(235.034976293715), 121.093143432996, 14.072583867850, 7.282670938482),
            q4.Strategy("FY3", q4.UAVS["FY3"], math.radians(76.171436922335), 134.227910409102, 23.726708210759, 1.592426235844),
        ]
    raise ValueError("no warm synergy candidate for this pair")


def pair_score(x: np.ndarray, names: list[str], dt: float, points: np.ndarray) -> float:
    total, _, synergy, _, near = pair_total_stats(decode_subset(x, names), dt, points)
    return -(total + 0.25 * synergy + 0.01 * near)


def pair_population(
    seed: int,
    names: list[str],
    baseline: list[q4.Strategy],
    synergy: list[q4.Strategy],
    size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pair_bounds = q4.BOUNDS_ONE * len(names)
    lower = np.array([a for a, _ in pair_bounds])
    upper = np.array([b for _, b in pair_bounds])
    population = rng.uniform(lower, upper, size=(size, len(lower)))
    scales = np.tile(np.array([math.radians(7.0), 8.0, 2.0, 0.10]), len(names))
    baseline_x, synergy_x = q4.encode(baseline), q4.encode(synergy)
    half = size // 2
    population[:half] = np.clip(
        baseline_x + rng.normal(size=(half, len(lower))) * scales,
        lower,
        upper,
    )
    population[half : half + size // 4] = np.clip(
        synergy_x + rng.normal(size=(size // 4, len(lower))) * scales,
        lower,
        upper,
    )
    population[0], population[half] = baseline_x, synergy_x
    return population


def optimize_pair_total(
    names: list[str], seeds: list[int], quick: bool
) -> tuple[list[q4.Strategy], list[tuple[int, float, float]]]:
    baseline_by_name = {strategy.name: strategy for strategy in q4.final_total_candidate()}
    baseline = [baseline_by_name[name] for name in names]
    synergy = known_synergy_candidate(names)
    pair_bounds = q4.BOUNDS_ONE * len(names)
    points = q4.surface_points(14 if quick else 24, 4 if quick else 7, 2 if quick else 4)
    candidates = [q4.encode(baseline), q4.encode(synergy)]
    runs: list[tuple[int, float, float]] = []
    for seed in seeds:
        size = 24 if quick else 88
        result = differential_evolution(
            lambda x: pair_score(x, names, 0.30 if quick else 0.12, points),
            pair_bounds,
            seed=seed,
            init=pair_population(seed, names, baseline, synergy, size),
            maxiter=12 if quick else 85,
            tol=3e-5,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidates.append(result.x.copy())
        total, _, syn, _, _ = pair_total_stats(
            decode_subset(result.x, names), 0.05, q4.surface_points(48, 11, 8)
        )
        runs.append((seed, total, syn))

    medium = q4.surface_points(64, 15, 10)
    best = min(candidates, key=lambda x: pair_score(x, names, 0.04, medium))
    refined = minimize(
        lambda x: pair_score(x, names, 0.025, medium),
        best,
        method="Nelder-Mead",
        bounds=pair_bounds,
        options={"maxiter": 120 if quick else 900, "xatol": 2e-8, "fatol": 2e-8},
    )
    return decode_subset(refined.x, names), runs


def bridge_total_stats(
    strategies: list[q4.Strategy], dt: float, points: np.ndarray
) -> dict[str, float | list[tuple[float, float]]]:
    """Evaluate the exact FY1--FY2--FY3 bridge decomposition.

    FY1 and FY3 cannot contribute at the same time, hence
    |S12 union S23| = |S12| + |S23| - |S2|.
    """
    by_name = {strategy.name: strategy for strategy in strategies}
    if set(by_name) != set(q4.UAVS):
        raise ValueError("bridge evaluation requires FY1, FY2, and FY3")
    pair12 = pair_total_stats([by_name["FY1"], by_name["FY2"]], dt, points)
    pair23 = pair_total_stats([by_name["FY2"], by_name["FY3"]], dt, points)
    fy2_total = q4.metrics([by_name["FY2"]], dt, points)[1]
    total = pair12[0] + pair23[0] - fy2_total
    return {
        "total": total,
        "pair12_total": pair12[0],
        "pair23_total": pair23[0],
        "fy2_total": fy2_total,
        "pair12_synergy": pair12[2],
        "pair23_synergy": pair23[2],
        "near": pair12[4] + pair23[4],
    }


def strategy_references(name: str) -> list[q4.Strategy]:
    baseline = {strategy.name: strategy for strategy in q4.final_total_candidate()}
    early = {strategy.name: strategy for strategy in known_synergy_candidate(["FY1", "FY2"])}
    late = {strategy.name: strategy for strategy in known_synergy_candidate(["FY2", "FY3"])}
    references = [baseline[name]]
    if name in early:
        references.append(early[name])
    if name in late:
        references.append(late[name])
    return references


def bridge_population(
    seed: int, current: q4.Strategy, references: list[q4.Strategy], size: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([a for a, _ in q4.BOUNDS_ONE])
    upper = np.array([b for _, b in q4.BOUNDS_ONE])
    population = rng.uniform(lower, upper, size=(size, 4))
    current_x = q4.encode([current])
    scales = np.array([math.radians(7.0), 8.0, 2.0, 0.10])
    local = size // 2
    population[:local] = np.clip(
        current_x + rng.normal(size=(local, 4)) * scales,
        lower,
        upper,
    )
    for index, reference in enumerate(references[: min(len(references), size)]):
        population[index] = np.clip(q4.encode([reference]), lower, upper)
    population[0] = np.clip(current_x, lower, upper)
    return population


def replace_strategy(
    strategies: list[q4.Strategy], name: str, x: np.ndarray
) -> list[q4.Strategy]:
    return [q4.decode_one(name, x) if strategy.name == name else strategy for strategy in strategies]


def bridge_score(
    x: np.ndarray,
    strategies: list[q4.Strategy],
    name: str,
    dt: float,
    points: np.ndarray,
    guided: bool,
) -> float:
    stats = bridge_total_stats(replace_strategy(strategies, name, x), dt, points)
    guide = 0.004 * float(stats["near"]) if guided else 0.0
    return -(float(stats["total"]) + guide)


def optimize_bridge_start(
    start: list[q4.Strategy], seed: int, quick: bool
) -> tuple[list[q4.Strategy], list[tuple[int, str, float]]]:
    current = list(start)
    coarse = q4.q2.ring_points(20 if quick else 28)
    audit = q4.surface_points(40 if quick else 64, 9 if quick else 15, 6 if quick else 10)
    audit_dt = 0.06 if quick else 0.035
    history: list[tuple[int, str, float]] = []
    for cycle in range(1 if quick else 2):
        for offset, name in enumerate(("FY1", "FY3", "FY2")):
            current_strategy = next(strategy for strategy in current if strategy.name == name)
            move_seed = seed + 1009 * cycle + 97 * offset
            result = differential_evolution(
                lambda x: bridge_score(
                    x,
                    current,
                    name,
                    0.24 if quick else 0.14,
                    coarse,
                    True,
                ),
                q4.BOUNDS_ONE,
                seed=move_seed,
                init=bridge_population(
                    move_seed,
                    current_strategy,
                    strategy_references(name),
                    24 if quick else 48,
                ),
                maxiter=14 if quick else 36,
                tol=5e-5,
                polish=False,
                workers=1,
                updating="immediate",
            )
            candidate = replace_strategy(current, name, result.x)
            old_total = float(bridge_total_stats(current, audit_dt, audit)["total"])
            new_total = float(bridge_total_stats(candidate, audit_dt, audit)["total"])
            if new_total >= old_total - 1e-8:
                current = candidate
                old_total = new_total
            history.append((cycle, name, old_total))
    return current, history


def optimize_bridge_total(
    seeds: list[int], quick: bool
) -> tuple[list[q4.Strategy], list[tuple[str, int, float, list[tuple[int, str, float]]]]]:
    baseline = q4.final_total_candidate()
    baseline_by_name = {strategy.name: strategy for strategy in baseline}
    early = {strategy.name: strategy for strategy in known_synergy_candidate(["FY1", "FY2"])}
    late = {strategy.name: strategy for strategy in known_synergy_candidate(["FY2", "FY3"])}
    starts = {
        "baseline": baseline,
        "early_pair": [early["FY1"], early["FY2"], baseline_by_name["FY3"]],
        "late_pair": [baseline_by_name["FY1"], late["FY2"], late["FY3"]],
    }
    audit = q4.surface_points(64, 15, 10)
    candidates = [baseline]
    runs: list[tuple[str, int, float, list[tuple[int, str, float]]]] = []
    for index, (label, start) in enumerate(starts.items()):
        for base_seed in seeds:
            seed = base_seed + 7919 * index
            candidate, history = optimize_bridge_start(start, seed, quick)
            total = float(bridge_total_stats(candidate, 0.025, audit)["total"])
            candidates.append(candidate)
            runs.append((label, seed, total, history))
    best = max(candidates, key=lambda item: float(bridge_total_stats(item, 0.025, audit)["total"]))
    return best, runs


def self_test() -> None:
    t = 15.0
    points = q4.surface_points(12, 3, 2)
    stats = responsibility(q4.final_total_candidate(), t, points)
    assert abs(float(np.sum(stats["hard_shares"])) - 1.0) < 1e-12
    assert abs(float(np.sum(stats["soft_shares"])) - 1.0) < 1e-12
    assert interval_difference([(0.0, 3.0)], [(1.0, 2.0)]) == [
        (0.0, 1.0),
        (2.0, 3.0),
    ]
    certificate = triple_reachability_certificate()
    assert not certificate["triple_exclusive_possible"]
    baseline = q4.final_total_candidate()
    test_points = q4.surface_points(16, 4, 3)
    bridge = float(bridge_total_stats(baseline, 0.10, test_points)["total"])
    direct = q4.metrics(baseline, 0.10, test_points)[1]
    # Separate sampled time grids introduce only interpolation-scale error.
    assert abs(bridge - direct) < 1e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="41,137")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--names", default="FY1,FY2")
    parser.add_argument("--time-range", default="9,15")
    parser.add_argument("--pair-total", action="store_true")
    parser.add_argument("--triple-audit", action="store_true")
    parser.add_argument("--bridge-total", action="store_true")
    args = parser.parse_args()
    self_test()
    names = args.names.split(",")
    if len(names) < 2 or any(name not in q4.UAVS for name in names):
        raise ValueError("--names must contain known UAV names")
    time_range = tuple(map(float, args.time_range.split(",")))
    if len(time_range) != 2 or not 0 <= time_range[0] < time_range[1]:
        raise ValueError("--time-range must be START,END")

    seeds = [int(item) for item in args.seeds.split(",")]
    if args.triple_audit:
        certificate = triple_reachability_certificate()
        print(f"triple_reachability_certificate={certificate}")
        for label, dt, points in (
            ("medium", 0.01, q4.surface_points(80, 19, 12)),
            ("dense", 0.002, q4.surface_points(128, 31, 18)),
        ):
            stats = coverage_order_stats(q4.final_total_candidate(), dt, points)
            print(f"{label}_coverage_order_stats={stats}")
        return

    if args.bridge_total:
        strategies, runs = optimize_bridge_total(seeds, args.quick)
        print(f"bridge_runs={runs}")
        print("bridge_best")
        for strategy in strategies:
            q4.print_strategy(strategy)
        for label, dt, points in (
            ("medium", 0.01, q4.surface_points(80, 19, 12)),
            ("dense", 0.002, q4.surface_points(128, 31, 18)),
        ):
            bridge = bridge_total_stats(strategies, dt, points)
            direct = q4.metrics(strategies, dt, points)
            print(f"{label}_bridge_stats={bridge}")
            print(f"{label}_direct_metrics={direct[:3]}")
        return

    if len(names) != 2:
        raise ValueError(
            "synergy search is pair-only; use --triple-audit for the certified "
            "three-ball coverage-order analysis"
        )

    if args.pair_total:
        if len(names) != 2:
            raise ValueError("--pair-total requires exactly two UAV names")
        strategies, runs = optimize_pair_total(names, seeds, args.quick)
        print(f"pair_total_runs={runs}")
        dense = pair_total_stats(strategies, 0.002, q4.surface_points(128, 31, 18))
        print(f"pair_total_dense={dense[:4]}")
        for strategy in strategies:
            q4.print_strategy(strategy)
        return

    audit_points = q4.surface_points(128, 31, 18)
    for seed in seeds:
        t, strategies, loss = search(seed, args.quick, names, time_range)
        stats = responsibility(strategies, t, audit_points)
        print(f"seed={seed} loss={loss:.12f} anchor_time={t:.12f}")
        print(f"joint_margin={stats['joint_margin']:.12f}")
        print(f"individual_margins={np.asarray(stats['individual_margins']).tolist()}")
        print(f"hard_shares={np.asarray(stats['hard_shares']).tolist()}")
        print(f"soft_shares={np.asarray(stats['soft_shares']).tolist()}")
        print(f"uncovered={stats['uncovered']:.12f} overlap={stats['overlap']:.12f}")
        for strategy in strategies:
            q4.print_strategy(strategy)


if __name__ == "__main__":
    main()
