"""CUMCM 2025 A/Q5 shared evaluator for three missiles and up to 15 bombs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

import q1_strict_occlusion as geometry
import q3_optimize as q3
import q4_optimize as q4


G = 9.8
MISSILES = {
    "M1": np.array([20_000.0, 0.0, 2_000.0]),
    "M2": np.array([19_000.0, 600.0, 2_100.0]),
    "M3": np.array([18_000.0, -600.0, 1_900.0]),
}
UAVS = {
    **q4.UAVS,
    "FY4": np.array([11_000.0, 2_000.0, 1_800.0]),
    "FY5": np.array([13_000.0, -2_000.0, 1_300.0]),
}
BOUNDS_ROUTE = [(0.0, 2 * math.pi), (70.0, 140.0)]
TRIAD_BOUNDS = BOUNDS_ROUTE + [(0.0, 1.0)] * 6


@dataclass(frozen=True)
class BombStrategy:
    uav: str
    number: int
    theta: float
    speed: float
    release_time: float
    delay: float
    assigned_missile: str
    gravity: float = G

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), 0.0])

    @property
    def explosion_time(self) -> float:
        return self.release_time + self.delay

    @property
    def release_point(self) -> np.ndarray:
        return UAVS[self.uav] + self.speed * self.release_time * self.direction

    @property
    def explosion_point(self) -> np.ndarray:
        point = UAVS[self.uav] + self.speed * self.explosion_time * self.direction
        point[2] -= 0.5 * self.gravity * self.delay**2
        return point

    def cloud_center(self, t: float) -> np.ndarray:
        return self.explosion_point - np.array(
            [0.0, 0.0, geometry.SMOKE_SINK_SPEED * (t - self.explosion_time)]
        )


def missile_hit_time(name: str) -> float:
    return float(np.linalg.norm(MISSILES[name]) / geometry.MISSILE_SPEED)


def decode_one(uav: str, missile: str, x: np.ndarray) -> BombStrategy:
    theta, speed, explosion_time, delay_fraction = map(float, x)
    free_fall = math.sqrt(2 * UAVS[uav][2] / G)
    delay = delay_fraction * min(explosion_time, free_fall)
    return BombStrategy(
        uav,
        1,
        theta % (2 * math.pi),
        speed,
        explosion_time - delay,
        delay,
        missile,
    )


def decode_triad(uav: str, missile: str, x: np.ndarray) -> list[BombStrategy]:
    """Decode one shared UAV route and three releases without constraint repair."""
    theta, speed = map(float, x[:2])
    limit = missile_hit_time(missile)
    release_1 = (limit - 2.0) * float(x[2]) ** 4
    release_2 = release_1 + 1.0 + (limit - release_1 - 2.0) * float(x[3]) ** 3
    release_3 = release_2 + 1.0 + (limit - release_2 - 1.0) * float(x[4]) ** 3
    releases = [release_1, release_2, release_3]
    free_fall = math.sqrt(2 * UAVS[uav][2] / G)
    delays = [
        min(free_fall, limit - release) * float(x[5 + i]) ** 2
        for i, release in enumerate(releases)
    ]
    return [
        BombStrategy(uav, i, theta % (2 * math.pi), speed, release, delay, missile)
        for i, (release, delay) in enumerate(zip(releases, delays), 1)
    ]


def decode_block(uav: str, label: str, x: np.ndarray) -> list[BombStrategy]:
    """Decode a refinement block on the full Q5 time horizon."""
    theta, speed = map(float, x[:2])
    limit = max(missile_hit_time(name) for name in MISSILES)
    release_1 = (limit - 2.0) * float(x[2]) ** 4
    release_2 = release_1 + 1.0 + (limit - release_1 - 2.0) * float(x[3]) ** 3
    release_3 = release_2 + 1.0 + (limit - release_2 - 1.0) * float(x[4]) ** 3
    releases = [release_1, release_2, release_3]
    free_fall = math.sqrt(2 * UAVS[uav][2] / G)
    delays = [
        min(free_fall, limit - release) * float(x[5 + i]) ** 2
        for i, release in enumerate(releases)
    ]
    return [
        BombStrategy(uav, i, theta % (2 * math.pi), speed, release, delay, label)
        for i, (release, delay) in enumerate(zip(releases, delays), 1)
    ]


def encode_block(strategies: list[BombStrategy]) -> np.ndarray:
    ordered = sorted(strategies, key=lambda item: item.release_time)
    limit = max(missile_hit_time(name) for name in MISSILES)
    releases = [item.release_time for item in ordered]
    x = np.zeros(8)
    x[0], x[1] = ordered[0].theta, ordered[0].speed
    x[2] = (releases[0] / (limit - 2.0)) ** 0.25
    x[3] = ((releases[1] - releases[0] - 1.0) / (limit - releases[0] - 2.0)) ** (1 / 3)
    x[4] = ((releases[2] - releases[1] - 1.0) / (limit - releases[1] - 1.0)) ** (1 / 3)
    free_fall = math.sqrt(2 * UAVS[ordered[0].uav][2] / G)
    for i, item in enumerate(ordered):
        cap = min(free_fall, limit - item.release_time)
        x[5 + i] = math.sqrt(item.delay / cap) if cap > 0 else 0.0
    lower = np.array([a for a, _ in TRIAD_BOUNDS])
    upper = np.array([b for _, b in TRIAD_BOUNDS])
    return np.clip(x, lower, upper)


def decode_mixed_block(
    uav: str,
    ordered_labels: list[str],
    x: np.ndarray,
) -> list[BombStrategy]:
    """Decode one UAV block while preserving the current release-order labels."""
    theta, speed = map(float, x[:2])
    count = len(ordered_labels)
    limit = max(missile_hit_time(name) for name in MISSILES)
    releases = [(limit - count + 1.0) * float(x[2]) ** 4]
    for index in range(1, count):
        previous = releases[-1]
        room = limit - previous - (count - index)
        releases.append(previous + 1.0 + room * float(x[2 + index]) ** 3)
    free_fall = math.sqrt(2 * UAVS[uav][2] / G)
    delays = [
        min(free_fall, limit - release) * float(x[2 + count + index]) ** 2
        for index, release in enumerate(releases)
    ]
    return [
        BombStrategy(uav, index, theta % (2 * math.pi), speed, release, delay, label)
        for index, (release, delay, label) in enumerate(
            zip(releases, delays, ordered_labels), 1
        )
    ]


def encode_mixed_block(strategies: list[BombStrategy]) -> tuple[np.ndarray, list[str]]:
    """Encode a one-to-three-bomb UAV block for mixed-missile local refinement."""
    ordered = sorted(strategies, key=lambda item: item.release_time)
    count = len(ordered)
    limit = max(missile_hit_time(name) for name in MISSILES)
    releases = [item.release_time for item in ordered]
    x = np.zeros(2 + 2 * count)
    x[0], x[1] = ordered[0].theta, ordered[0].speed
    x[2] = (releases[0] / (limit - count + 1.0)) ** 0.25
    for index in range(1, count):
        room = limit - releases[index - 1] - (count - index)
        fraction = (releases[index] - releases[index - 1] - 1.0) / room
        x[2 + index] = max(0.0, fraction) ** (1 / 3)
    free_fall = math.sqrt(2 * UAVS[ordered[0].uav][2] / G)
    for index, item in enumerate(ordered):
        cap = min(free_fall, limit - item.release_time)
        x[2 + count + index] = math.sqrt(item.delay / cap) if cap > 0 else 0.0
    bounds = BOUNDS_ROUTE + [(0.0, 1.0)] * (2 * count)
    lower = np.array([a for a, _ in bounds])
    upper = np.array([b for _, b in bounds])
    labels = [item.assigned_missile for item in ordered]
    return np.clip(x, lower, upper), labels


def encode_triad(strategies: list[BombStrategy], missile: str) -> np.ndarray:
    """Inverse of decode_triad for feasible warm starts."""
    ordered = sorted(strategies, key=lambda item: item.release_time)
    limit = missile_hit_time(missile)
    releases = [item.release_time for item in ordered]
    x = np.zeros(8)
    x[0], x[1] = ordered[0].theta, ordered[0].speed
    x[2] = (releases[0] / (limit - 2.0)) ** 0.25
    x[3] = ((releases[1] - releases[0] - 1.0) / (limit - releases[0] - 2.0)) ** (1 / 3)
    x[4] = ((releases[2] - releases[1] - 1.0) / (limit - releases[1] - 1.0)) ** (1 / 3)
    free_fall = math.sqrt(2 * UAVS[ordered[0].uav][2] / G)
    for i, item in enumerate(ordered):
        cap = min(free_fall, limit - item.release_time)
        x[5 + i] = math.sqrt(item.delay / cap) if cap > 0 else 0.0
    lower = np.array([a for a, _ in TRIAD_BOUNDS])
    upper = np.array([b for _, b in TRIAD_BOUNDS])
    return np.clip(x, lower, upper)


def warm_triad(single: BombStrategy, missile: str) -> list[BombStrategy]:
    """Turn a single-bomb optimum into a legal, deliberately simple relay seed."""
    if single.uav == "FY1" and missile == "M1":
        return [
            BombStrategy(
                "FY1",
                i,
                item.theta,
                item.speed,
                item.release_time,
                item.delay,
                "M1",
            )
            for i, item in enumerate(q3.INCUMBENT, 1)
        ]
    limit = missile_hit_time(missile)
    gap = min(8.0, (limit - 2.0) / 3.0)
    start = min(single.release_time, limit - 2.0 * gap)
    releases = [start, start + gap, start + 2.0 * gap]
    free_fall = math.sqrt(2 * UAVS[single.uav][2] / G)
    return [
        BombStrategy(
            single.uav,
            i,
            single.theta,
            single.speed,
            release,
            min(single.delay, free_fall, limit - release),
            missile,
        )
        for i, release in enumerate(releases, 1)
    ]


def validate(strategies: list[BombStrategy]) -> None:
    for strategy in strategies:
        assert strategy.uav in UAVS and strategy.assigned_missile in MISSILES
        assert 1 <= strategy.number <= 3
        assert 70.0 <= strategy.speed <= 140.0
        assert 0.0 <= strategy.release_time <= strategy.explosion_time
        assert strategy.explosion_point[2] >= -1e-9
    for name in UAVS:
        group = sorted(
            (strategy for strategy in strategies if strategy.uav == name),
            key=lambda strategy: strategy.release_time,
        )
        assert len(group) <= 3
        if not group:
            continue
        assert all(abs(item.theta - group[0].theta) <= 1e-10 for item in group)
        assert all(abs(item.speed - group[0].speed) <= 1e-10 for item in group)
        assert all(
            right.release_time - left.release_time >= 1.0 - 1e-10
            for left, right in zip(group, group[1:])
        )


def time_grid(strategies: list[BombStrategy], missile: str, dt: float) -> np.ndarray:
    hit = missile_hit_time(missile)
    starts = [item.explosion_time for item in strategies if item.explosion_time <= hit]
    if not starts:
        return np.array([], dtype=float)
    start = min(starts)
    end = min(hit, max(item.explosion_time + geometry.SMOKE_LIFETIME for item in strategies))
    events = [
        value
        for item in strategies
        for value in (item.explosion_time, min(item.explosion_time + geometry.SMOKE_LIFETIME, hit))
        if start <= value <= end
    ]
    regular = np.arange(start, end + dt / 2, dt)
    return np.unique(np.clip(np.r_[regular, events, end], start, end))


def joint_worst_distances(
    strategies: list[BombStrategy], missile: str, times: np.ndarray, points: np.ndarray
) -> np.ndarray:
    if len(times) == 0:
        return np.array([], dtype=float)
    missile0 = MISSILES[missile]
    direction = -missile0 / np.linalg.norm(missile0)
    missiles = missile0 + geometry.MISSILE_SPEED * times[:, None] * direction
    worst = np.empty(len(times), dtype=float)
    for start in range(0, len(times), 64):
        stop = min(start + 64, len(times))
        chunk_times, chunk_missiles = times[start:stop], missiles[start:stop]
        best = np.full((len(chunk_times), len(points)), np.inf)
        for strategy in strategies:
            active = (
                (chunk_times >= strategy.explosion_time - 1e-12)
                & (chunk_times <= strategy.explosion_time + geometry.SMOKE_LIFETIME + 1e-12)
            )
            if not np.any(active):
                continue
            active_times = chunk_times[active]
            active_missiles = chunk_missiles[active]
            clouds = np.stack([strategy.cloud_center(float(t)) for t in active_times])
            segments = points[None, :, :] - active_missiles[:, None, :]
            offsets = clouds - active_missiles
            lam = np.einsum("tpi,ti->tp", segments, offsets) / np.einsum(
                "tpi,tpi->tp", segments, segments
            )
            lam = np.clip(lam, 0.0, 1.0)
            closest = active_missiles[:, None, :] + lam[:, :, None] * segments
            distances = np.linalg.norm(closest - clouds[:, None, :], axis=2)
            best[active] = np.minimum(best[active], distances)
        worst[start:stop] = np.max(best, axis=1)
    return worst


def missile_intervals(
    strategies: list[BombStrategy], missile: str, dt: float, points: np.ndarray
) -> list[tuple[float, float]]:
    times = time_grid(strategies, missile, dt)
    if len(times) == 0:
        return []
    excess = joint_worst_distances(strategies, missile, times, points) - geometry.SMOKE_RADIUS
    return q4.sampled_intervals(times, excess)


def intersect_intervals(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def metrics(
    strategies: list[BombStrategy], dt: float, points: np.ndarray
) -> dict[str, object]:
    validate(strategies)
    intervals = {
        name: missile_intervals(strategies, name, dt, points) for name in MISSILES
    }
    durations = {
        name: sum(end - start for start, end in current)
        for name, current in intervals.items()
    }
    simultaneous = list(intervals.values())[0]
    for current in list(intervals.values())[1:]:
        simultaneous = intersect_intervals(simultaneous, current)
    return {
        "intervals": intervals,
        "durations": durations,
        "total": sum(durations.values()),
        "minimum": min(durations.values()),
        "simultaneous_intervals": simultaneous,
        "simultaneous": sum(end - start for start, end in simultaneous),
    }


def metrics_on_grids(
    strategies: list[BombStrategy],
    grids: dict[str, np.ndarray],
    points: np.ndarray,
) -> dict[str, object]:
    validate(strategies)
    intervals = {}
    for missile, times in grids.items():
        excess = joint_worst_distances(strategies, missile, times, points) - geometry.SMOKE_RADIUS
        intervals[missile] = q4.sampled_intervals(times, excess)
    durations = {
        name: sum(end - start for start, end in current)
        for name, current in intervals.items()
    }
    simultaneous = list(intervals.values())[0]
    for current in list(intervals.values())[1:]:
        simultaneous = intersect_intervals(simultaneous, current)
    return {
        "intervals": intervals,
        "durations": durations,
        "total": sum(durations.values()),
        "minimum": min(durations.values()),
        "simultaneous_intervals": simultaneous,
        "simultaneous": sum(end - start for start, end in simultaneous),
    }


def single_metrics(
    strategy: BombStrategy, missile: str, dt: float, points: np.ndarray
) -> tuple[float, list[tuple[float, float]], float]:
    times = time_grid([strategy], missile, dt)
    excess = joint_worst_distances([strategy], missile, times, points) - geometry.SMOKE_RADIUS
    intervals = q4.sampled_intervals(times, excess)
    duration = sum(end - start for start, end in intervals)
    return duration, intervals, float(np.min(excess))


def single_score(
    x: np.ndarray, uav: str, missile: str, dt: float, points: np.ndarray, guided: bool
) -> float:
    duration, _, minimum = single_metrics(decode_one(uav, missile, x), missile, dt, points)
    guide = 0.30 * math.exp(-max(0.0, minimum) / 12.0) if guided else 0.0
    return -(duration + guide)


def single_population(uav: str, missile: str, seed: int, size: int) -> np.ndarray:
    bounds = BOUNDS_ROUTE + [(0.0, missile_hit_time(missile)), (0.0, 1.0)]
    rng = np.random.default_rng(seed)
    population = rng.uniform(
        [low for low, _ in bounds], [high for _, high in bounds], size=(size, 4)
    )
    if missile == "M1" and uav in q4.UAVS:
        known = {strategy.name: strategy for strategy in q4.final_total_candidate()}[uav]
        population[0] = q4.encode([known])
    return population


def optimize_single(uav: str, missile: str, seed: int, quick: bool) -> BombStrategy:
    bounds = BOUNDS_ROUTE + [(0.0, missile_hit_time(missile)), (0.0, 1.0)]
    coarse = q4.q2.ring_points(16 if quick else 24)
    result = differential_evolution(
        lambda x: single_score(x, uav, missile, 0.30 if quick else 0.18, coarse, True),
        bounds,
        seed=seed,
        init=single_population(uav, missile, seed, 28 if quick else 48),
        maxiter=35 if quick else 75,
        tol=8e-5,
        polish=False,
        workers=1,
        updating="immediate",
    )
    medium = q4.q2.ring_points(64)
    refined = minimize(
        lambda x: single_score(x, uav, missile, 0.05, medium, False),
        result.x,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 120 if quick else 350, "xatol": 2e-7, "fatol": 2e-7},
    )
    return decode_one(uav, missile, refined.x)


def triad_components(
    x: np.ndarray,
    uav: str,
    missile: str,
    dt: float,
    points: np.ndarray,
) -> tuple[float, list[tuple[float, float]], float]:
    strategies = decode_triad(uav, missile, x)
    times = time_grid(strategies, missile, dt)
    excess = joint_worst_distances(strategies, missile, times, points) - geometry.SMOKE_RADIUS
    intervals = q4.sampled_intervals(times, excess)
    duration = sum(end - start for start, end in intervals)
    near = np.zeros_like(times)
    outside = np.isfinite(excess) & (excess > 0.0)
    near[outside] = np.exp(-excess[outside] / 2.5)
    guide = float(np.trapezoid(near, times)) if len(times) > 1 else 0.0
    return duration, intervals, guide


def triad_score(
    x: np.ndarray,
    uav: str,
    missile: str,
    dt: float,
    points: np.ndarray,
    guided: bool,
) -> float:
    duration, _, near = triad_components(x, uav, missile, dt, points)
    return -(duration + (0.01 * near if guided else 0.0))


def triad_population(
    uav: str,
    missile: str,
    single: BombStrategy,
    seed: int,
    size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([a for a, _ in TRIAD_BOUNDS])
    upper = np.array([b for _, b in TRIAD_BOUNDS])
    population = rng.uniform(lower, upper, size=(size, 8))
    warm = encode_triad(warm_triad(single, missile), missile)
    local = size // 3
    scales = np.array(
        [math.radians(1.5), 4.0, 0.04, 0.06, 0.06, 0.08, 0.08, 0.08]
    )
    population[:local] = np.clip(
        warm + rng.normal(size=(local, 8)) * scales,
        lower,
        upper,
    )
    population[local : 2 * local, :2] = np.clip(
        warm[:2] + rng.normal(size=(local, 2)) * scales[:2],
        lower[:2],
        upper[:2],
    )
    population[0] = warm
    return population


def optimize_triad(
    uav: str,
    missile: str,
    seeds: list[int],
    quick: bool,
) -> tuple[list[BombStrategy], list[dict[str, float]]]:
    single = optimize_single(uav, missile, seeds[0], quick)
    warm = encode_triad(warm_triad(single, missile), missile)
    coarse = q4.surface_points(16 if quick else 24, 4 if quick else 7, 3 if quick else 4)
    candidates = [warm]
    runs: list[dict[str, float]] = []
    for seed in seeds:
        result = differential_evolution(
            lambda x: triad_score(
                x,
                uav,
                missile,
                0.24 if quick else 0.14,
                coarse,
                True,
            ),
            TRIAD_BOUNDS,
            seed=seed,
            init=triad_population(uav, missile, single, seed, 48 if quick else 96),
            maxiter=25 if quick else 80,
            tol=4e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidates.append(result.x.copy())
        runs.append({"seed": seed, "guided_score": -float(result.fun)})

    audit = q4.surface_points(32 if quick else 64, 7 if quick else 15, 5 if quick else 10)
    best = min(
        candidates,
        key=lambda x: triad_score(x, uav, missile, 0.06 if quick else 0.035, audit, False),
    )
    refined = minimize(
        lambda x: triad_score(
            x,
            uav,
            missile,
            0.04 if quick else 0.015,
            audit,
            False,
        ),
        best,
        method="Nelder-Mead",
        bounds=TRIAD_BOUNDS,
        options={"maxiter": 120 if quick else 700, "xatol": 2e-8, "fatol": 2e-8},
    ).x
    winner = min(
        [warm, best, refined],
        key=lambda x: triad_score(x, uav, missile, 0.03 if quick else 0.015, audit, False),
    )
    return decode_triad(uav, missile, winner), runs


def strategy_record(strategy: BombStrategy) -> dict[str, object]:
    return {
        "uav": strategy.uav,
        "number": strategy.number,
        "theta_rad": strategy.theta,
        "theta_deg": math.degrees(strategy.theta),
        "speed": strategy.speed,
        "release_time": strategy.release_time,
        "release_point": strategy.release_point.tolist(),
        "delay": strategy.delay,
        "explosion_time": strategy.explosion_time,
        "explosion_point": strategy.explosion_point.tolist(),
        "assigned_missile": strategy.assigned_missile,
    }


def strategy_from_record(record: dict[str, object]) -> BombStrategy:
    return BombStrategy(
        str(record["uav"]),
        int(record["number"]),
        float(record["theta_rad"]),
        float(record["speed"]),
        float(record["release_time"]),
        float(record["delay"]),
        str(record["assigned_missile"]),
    )


def save_triad_candidate(
    output: Path,
    uav: str,
    missile: str,
    strategies: list[BombStrategy],
    runs: list[dict[str, float]],
    quick: bool,
) -> dict[str, object]:
    if output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "gravity": G, "candidates": {}}
    audit_points = q4.surface_points(64, 15, 10)
    result = metrics(strategies, 0.02, audit_points)
    focus_duration = float(result["durations"][missile])
    key = f"{uav}-{missile}"
    data["candidates"][key] = {
        "focus_uav": uav,
        "focus_missile": missile,
        "quick_search": quick,
        "focus_duration": focus_duration,
        "all_missile_metrics": result,
        "runs": runs,
        "bombs": [strategy_record(item) for item in strategies],
    }
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data["candidates"][key]


def enumerate_candidate_combinations(
    candidate_file: Path,
    output: Path,
    dt: float,
    points: np.ndarray,
) -> list[dict[str, object]]:
    library = json.loads(candidate_file.read_text(encoding="utf-8"))["candidates"]
    missing = [
        f"{uav}-{missile}"
        for uav in UAVS
        for missile in MISSILES
        if f"{uav}-{missile}" not in library
    ]
    if missing:
        raise ValueError(f"candidate library is incomplete: {missing}")
    ranked: list[dict[str, object]] = []
    uavs = list(UAVS)
    for assignment in itertools.product(MISSILES, repeat=len(uavs)):
        strategies = [
            strategy_from_record(record)
            for uav, missile in zip(uavs, assignment)
            for record in library[f"{uav}-{missile}"]["bombs"]
        ]
        result = metrics(strategies, dt, points)
        ranked.append(
            {
                "assignment": dict(zip(uavs, assignment)),
                "metrics": result,
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item["metrics"]["total"]),
            float(item["metrics"]["minimum"]),
            float(item["metrics"]["simultaneous"]),
        ),
        reverse=True,
    )
    output.write_text(
        json.dumps(
            {
                "candidate_file": str(candidate_file.resolve()),
                "dt": dt,
                "surface_point_count": len(points),
                "ranking": ranked,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ranked


def audit_top_combinations(
    candidate_file: Path,
    ranking_file: Path,
    output: Path,
    top: int,
    dt: float,
    points: np.ndarray,
) -> list[dict[str, object]]:
    library = json.loads(candidate_file.read_text(encoding="utf-8"))["candidates"]
    screening = json.loads(ranking_file.read_text(encoding="utf-8"))["ranking"]
    audited: list[dict[str, object]] = []
    for item in screening[:top]:
        assignment = item["assignment"]
        strategies = [
            strategy_from_record(record)
            for uav in UAVS
            for record in library[f"{uav}-{assignment[uav]}"]["bombs"]
        ]
        audited.append({"assignment": assignment, "metrics": metrics(strategies, dt, points)})
    audited.sort(
        key=lambda item: (
            float(item["metrics"]["total"]),
            float(item["metrics"]["minimum"]),
            float(item["metrics"]["simultaneous"]),
        ),
        reverse=True,
    )
    output.write_text(
        json.dumps(
            {
                "candidate_file": str(candidate_file.resolve()),
                "screening_file": str(ranking_file.resolve()),
                "dt": dt,
                "surface_point_count": len(points),
                "ranking": audited,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return audited


def refine_assignment(
    candidate_file: Path,
    audit_file: Path,
    rank: int,
    output: Path,
    cycles: int,
    quick: bool,
    only_uav: str | None,
) -> dict[str, object]:
    library = json.loads(candidate_file.read_text(encoding="utf-8"))["candidates"]
    leaders = json.loads(audit_file.read_text(encoding="utf-8"))["ranking"]
    assignment = leaders[rank - 1]["assignment"]
    strategies = [
        strategy_from_record(record)
        for uav in UAVS
        for record in library[f"{uav}-{assignment[uav]}"]["bombs"]
    ]
    history: list[dict[str, object]] = []
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        prior = next(
            (item for item in saved.get("plans", []) if item.get("assignment") == assignment),
            None,
        )
        if prior is not None:
            strategies = [strategy_from_record(item) for item in prior["bombs"]]
            history = list(prior.get("history", []))
    search_points = q4.surface_points(16 if quick else 24, 4 if quick else 7, 3 if quick else 4)
    acceptance_points = q4.surface_points(32 if quick else 48, 7 if quick else 11, 5 if quick else 8)
    start_cycle = max((int(item["cycle"]) for item in history), default=0) + 1
    for cycle in range(start_cycle, start_cycle + cycles):
        changed = False
        for uav in ([only_uav] if only_uav else UAVS):
            current_group = [item for item in strategies if item.uav == uav]
            fixed = [item for item in strategies if item.uav != uav]
            label = str(assignment[uav])
            warm = encode_block(current_group)
            before = metrics(strategies, 0.05, acceptance_points)
            result = minimize(
                lambda x: -float(
                    metrics(
                        fixed + decode_block(uav, label, x),
                        0.12 if quick else 0.07,
                        search_points,
                    )["total"]
                ),
                warm,
                method="Nelder-Mead",
                bounds=TRIAD_BOUNDS,
                options={
                    "maxiter": 90 if quick else 220,
                    "xatol": 2e-7,
                    "fatol": 2e-7,
                },
            )
            candidate = fixed + decode_block(uav, label, result.x)
            after = metrics(candidate, 0.05, acceptance_points)
            accepted = float(after["total"]) > float(before["total"]) + 1e-6
            if accepted:
                strategies = candidate
                changed = True
            history.append(
                {
                    "cycle": cycle,
                    "uav": uav,
                    "accepted": accepted,
                    "before_total": before["total"],
                    "candidate_total": after["total"],
                }
            )
            print(
                f"cycle={cycle},uav={uav},accepted={accepted},"
                f"before={before['total']:.9f},candidate={after['total']:.9f}",
                flush=True,
            )
        if not changed:
            break
    final = metrics(strategies, 0.02, q4.surface_points(64, 15, 10))
    record = {
        "source_rank": rank,
        "assignment": assignment,
        "quick_refinement": quick,
        "cycles_completed": max((int(item["cycle"]) for item in history), default=0),
        "history": history,
        "metrics": final,
        "bombs": [strategy_record(item) for item in sorted(strategies, key=lambda x: (x.uav, x.number))],
    }
    if output.exists():
        data = json.loads(output.read_text(encoding="utf-8"))
    else:
        data = {"version": 1, "plans": []}
    data["plans"] = [
        item for item in data["plans"] if item.get("assignment") != assignment
    ] + [record]
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def recombine_refined_plans(
    refinement_file: Path,
    output: Path,
) -> list[dict[str, object]]:
    plans = json.loads(refinement_file.read_text(encoding="utf-8"))["plans"]
    if len(plans) < 2:
        raise ValueError("at least two refined plans are required")
    blocks = [
        {
            uav: [
                strategy_from_record(item)
                for item in plan["bombs"]
                if item["uav"] == uav
            ]
            for uav in UAVS
        }
        for plan in plans
    ]
    screening_points = q4.surface_points(32, 7, 5)
    screened: list[dict[str, object]] = []
    for choices in itertools.product(range(len(plans)), repeat=len(UAVS)):
        strategies = [
            item
            for uav, source in zip(UAVS, choices)
            for item in blocks[source][uav]
        ]
        screened.append(
            {
                "source_plan_by_uav": dict(zip(UAVS, choices)),
                "metrics": metrics(strategies, 0.05, screening_points),
                "bombs": [
                    strategy_record(item)
                    for item in sorted(strategies, key=lambda x: (x.uav, x.number))
                ],
            }
        )
    screened.sort(key=lambda item: float(item["metrics"]["total"]), reverse=True)
    audit_points = q4.surface_points(64, 15, 10)
    audited: list[dict[str, object]] = []
    for item in screened[:10]:
        strategies = [strategy_from_record(record) for record in item["bombs"]]
        audited.append(
            {
                "source_plan_by_uav": item["source_plan_by_uav"],
                "metrics": metrics(strategies, 0.02, audit_points),
                "bombs": item["bombs"],
            }
        )
    audited.sort(
        key=lambda item: (
            float(item["metrics"]["total"]),
            float(item["metrics"]["minimum"]),
            float(item["metrics"]["simultaneous"]),
        ),
        reverse=True,
    )
    output.write_text(
        json.dumps({"version": 1, "plans": audited}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audited


def validate_balanced_plan(
    refinement_file: Path,
    output: Path,
    plan_file: Path | None = None,
) -> dict[str, object]:
    if plan_file is None:
        plans = json.loads(refinement_file.read_text(encoding="utf-8"))["plans"]
        feasible = [item for item in plans if float(item["metrics"]["minimum"]) > 0.0]
        if not feasible:
            raise ValueError("no saved plan effectively interferes with all three missiles")
        plan = max(feasible, key=lambda item: float(item["metrics"]["total"]))
        strategies = [strategy_from_record(item) for item in plan["bombs"]]
        assignment = plan["assignment"]
    else:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        strategies = [strategy_from_record(item) for item in plan["bombs"]]
        assignment = {uav: sorted({item.assigned_missile for item in strategies if item.uav == uav}) for uav in UAVS}
    validate(strategies)

    resolutions = {}
    for name, dt, points in (
        ("coarse", 0.04, q4.surface_points(48, 11, 8)),
        ("medium", 0.02, q4.surface_points(64, 15, 10)),
        ("fine", 0.01, q4.surface_points(80, 19, 12)),
        ("dense", 0.005, q4.surface_points(96, 21, 14)),
    ):
        resolutions[name] = {
            "dt": dt,
            "surface_point_count": len(points),
            "metrics": metrics(strategies, dt, points),
        }
        print(
            f"validation={name},total={resolutions[name]['metrics']['total']:.9f}",
            flush=True,
        )

    marginal_points = q4.surface_points(64, 15, 10)
    grids = {name: time_grid(strategies, name, 0.02) for name in MISSILES}
    base = metrics_on_grids(strategies, grids, marginal_points)
    marginals = []
    for removed in strategies:
        reduced = metrics_on_grids(
            [item for item in strategies if item is not removed], grids, marginal_points
        )
        contribution = {
            name: float(base["durations"][name]) - float(reduced["durations"][name])
            for name in MISSILES
        }
        marginals.append(
            {
                "uav": removed.uav,
                "number": removed.number,
                "current_label": removed.assigned_missile,
                "duration_loss_by_missile": contribution,
                "total_loss": sum(contribution.values()),
            }
        )

    rounded = [
        BombStrategy(
            item.uav,
            item.number,
            math.radians(round(math.degrees(item.theta), 3)),
            round(item.speed, 3),
            round(item.release_time, 3),
            round(item.delay, 3),
            item.assigned_missile,
        )
        for item in strategies
    ]
    rounding = metrics(rounded, 0.02, marginal_points)
    gravity = {}
    for value in (9.79, 9.80, 9.81):
        perturbed = [
            BombStrategy(
                item.uav,
                item.number,
                item.theta,
                item.speed,
                item.release_time,
                item.delay,
                item.assigned_missile,
                value,
            )
            for item in strategies
        ]
        gravity[f"{value:.2f}"] = metrics(perturbed, 0.02, marginal_points)

    record = {
        "assignment": assignment,
        "resolutions": resolutions,
        "fixed_grid_base": base,
        "marginal_contributions": marginals,
        "rounded_decision_metrics": rounding,
        "gravity_sensitivity": gravity,
        "bombs": [strategy_record(item) for item in strategies],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def decode_fixed_route_bomb(
    uav: str,
    label: str,
    theta: float,
    speed: float,
    number: int,
    x: np.ndarray,
) -> BombStrategy:
    release, delay_fraction = map(float, x)
    cap = min(math.sqrt(2 * UAVS[uav][2] / G), missile_hit_time(label) - release)
    return BombStrategy(uav, number, theta, speed, release, max(0.0, cap) * delay_fraction, label)


def fixed_route_bomb_score(
    x: np.ndarray,
    active: list[BombStrategy],
    uav: str,
    label: str,
    theta: float,
    speed: float,
    number: int,
    dt: float,
    points: np.ndarray,
) -> float:
    release = float(x[0])
    same_uav = [item for item in active if item.uav == uav]
    gap = min((abs(release - item.release_time) for item in same_uav), default=math.inf)
    if gap < 1.0:
        return 100.0 + 100.0 * (1.0 - gap)
    candidate = decode_fixed_route_bomb(uav, label, theta, speed, number, x)
    intervals = missile_intervals(active + [candidate], label, dt, points)
    duration = sum(end - start for start, end in intervals)
    _, _, minimum = single_metrics(candidate, label, dt, points)
    guide = 0.05 * math.exp(-max(0.0, minimum) / 10.0)
    return -(duration + guide)


def optimize_fixed_route_bomb(
    active: list[BombStrategy],
    uav: str,
    label: str,
    seed: int,
    quick: bool,
) -> BombStrategy:
    group = [item for item in active if item.uav == uav]
    theta, speed = group[0].theta, group[0].speed
    number = len(group) + 1
    bounds = [(0.0, missile_hit_time(label)), (0.0, 1.0)]
    coarse = q4.surface_points(16 if quick else 24, 4 if quick else 7, 3 if quick else 4)
    result = differential_evolution(
        lambda x: fixed_route_bomb_score(
            x,
            active,
            uav,
            label,
            theta,
            speed,
            number,
            0.20 if quick else 0.12,
            coarse,
        ),
        bounds,
        seed=seed,
        popsize=10,
        maxiter=20 if quick else 45,
        tol=5e-4,
        polish=False,
        workers=1,
        updating="immediate",
    )
    audit = q4.surface_points(32 if quick else 48, 7 if quick else 11, 5 if quick else 8)
    refined = minimize(
        lambda x: fixed_route_bomb_score(
            x,
            active,
            uav,
            label,
            theta,
            speed,
            number,
            0.06 if quick else 0.035,
            audit,
        ),
        result.x,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 80 if quick else 180, "xatol": 2e-7, "fatol": 2e-7},
    )
    candidates = [result.x, refined.x]
    best = min(
        candidates,
        key=lambda x: fixed_route_bomb_score(
            x, active, uav, label, theta, speed, number, 0.04, audit
        ),
    )
    return decode_fixed_route_bomb(uav, label, theta, speed, number, best)


def activate_unused_bombs(
    validation_file: Path,
    output: Path,
    quick: bool,
) -> dict[str, object]:
    validated = json.loads(validation_file.read_text(encoding="utf-8"))
    strategies = [strategy_from_record(item) for item in validated["bombs"]]
    useful = {
        (item["uav"], int(item["number"]))
        for item in validated["marginal_contributions"]
        if float(item["total_loss"]) > 1e-6
    }
    active = [item for item in strategies if (item.uav, item.number) in useful]
    acceptance = q4.surface_points(48, 11, 8)
    history: list[dict[str, object]] = []
    for uav_index, uav in enumerate(UAVS):
        while sum(item.uav == uav for item in active) < 3:
            before = metrics(active, 0.04, acceptance)
            proposals = []
            for missile_index, missile in enumerate(MISSILES):
                candidate = optimize_fixed_route_bomb(
                    active,
                    uav,
                    missile,
                    701 + 101 * uav_index + 17 * missile_index + len(active),
                    quick,
                )
                proposal = active + [candidate]
                result = metrics(proposal, 0.04, acceptance)
                proposals.append((float(result["total"]), candidate, result))
            total, candidate, result = max(proposals, key=lambda item: item[0])
            improvement = total - float(before["total"])
            accepted = improvement > 0.01
            history.append(
                {
                    "uav": uav,
                    "slot": sum(item.uav == uav for item in active) + 1,
                    "accepted": accepted,
                    "target": candidate.assigned_missile,
                    "improvement": improvement,
                    "candidate": strategy_record(candidate),
                }
            )
            print(
                f"activate_uav={uav},slot={history[-1]['slot']},target={candidate.assigned_missile},"
                f"accepted={accepted},improvement={improvement:.9f}",
                flush=True,
            )
            if not accepted:
                break
            active.append(candidate)

    renumbered = []
    for uav in UAVS:
        for number, item in enumerate(
            sorted((current for current in active if current.uav == uav), key=lambda x: x.release_time),
            1,
        ):
            renumbered.append(
                BombStrategy(
                    item.uav,
                    number,
                    item.theta,
                    item.speed,
                    item.release_time,
                    item.delay,
                    item.assigned_missile,
                    item.gravity,
                )
            )
    final = metrics(renumbered, 0.01, q4.surface_points(80, 19, 12))
    record = {
        "source": str(validation_file.resolve()),
        "history": history,
        "metrics": final,
        "bombs": [strategy_record(item) for item in renumbered],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def attack_empty_slots(
    plan_file: Path,
    output: Path,
    seeds: list[int],
    quick: bool,
) -> dict[str, object]:
    source = json.loads(plan_file.read_text(encoding="utf-8"))
    active = [strategy_from_record(item) for item in source["bombs"]]
    acceptance = q4.surface_points(48, 11, 8)
    history: list[dict[str, object]] = []
    for uav in UAVS:
        while sum(item.uav == uav for item in active) < 3:
            before = metrics(active, 0.04, acceptance)
            proposals = []
            for missile in MISSILES:
                for seed in seeds:
                    candidate = optimize_fixed_route_bomb(
                        active, uav, missile, seed, quick
                    )
                    result = metrics(active + [candidate], 0.04, acceptance)
                    proposals.append((float(result["total"]), candidate, result, seed))
            total, candidate, result, seed = max(proposals, key=lambda item: item[0])
            improvement = total - float(before["total"])
            accepted = improvement > 0.005
            history.append(
                {
                    "uav": uav,
                    "slot": sum(item.uav == uav for item in active) + 1,
                    "accepted": accepted,
                    "target": candidate.assigned_missile,
                    "seed": seed,
                    "improvement": improvement,
                    "candidate": strategy_record(candidate),
                }
            )
            print(
                f"attack_uav={uav},slot={history[-1]['slot']},target={candidate.assigned_missile},"
                f"seed={seed},accepted={accepted},improvement={improvement:.9f}",
                flush=True,
            )
            if not accepted:
                break
            active.append(candidate)

    renumbered = []
    for uav in UAVS:
        group = sorted(
            (item for item in active if item.uav == uav),
            key=lambda item: item.release_time,
        )
        for number, item in enumerate(group, 1):
            renumbered.append(
                BombStrategy(
                    item.uav,
                    number,
                    item.theta,
                    item.speed,
                    item.release_time,
                    item.delay,
                    item.assigned_missile,
                    item.gravity,
                )
            )
    final = metrics(renumbered, 0.01, q4.surface_points(80, 19, 12))
    record = {
        "source": str(plan_file.resolve()),
        "seeds": seeds,
        "history": history,
        "metrics": final,
        "bombs": [strategy_record(item) for item in renumbered],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def prune_zero_marginal_bombs(
    validation_file: Path,
    output: Path,
    threshold: float,
) -> dict[str, object]:
    """Remove bombs whose fixed-grid leave-one-out loss does not exceed threshold."""
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    useful = {
        (item["uav"], int(item["number"]))
        for item in validation["marginal_contributions"]
        if float(item["total_loss"]) > threshold
    }
    removed = [
        item
        for item in validation["marginal_contributions"]
        if (item["uav"], int(item["number"])) not in useful
    ]
    active = [
        strategy_from_record(item)
        for item in validation["bombs"]
        if (item["uav"], int(item["number"])) in useful
    ]
    renumbered = []
    for uav in UAVS:
        group = sorted(
            (item for item in active if item.uav == uav),
            key=lambda item: item.release_time,
        )
        for number, item in enumerate(group, 1):
            renumbered.append(
                BombStrategy(
                    item.uav,
                    number,
                    item.theta,
                    item.speed,
                    item.release_time,
                    item.delay,
                    item.assigned_missile,
                    item.gravity,
                )
            )
    validate(renumbered)
    record = {
        "source": str(validation_file.resolve()),
        "marginal_threshold": threshold,
        "removed": removed,
        "metrics": metrics(renumbered, 0.01, q4.surface_points(80, 19, 12)),
        "bombs": [strategy_record(item) for item in renumbered],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def refine_active_blocks(
    plan_file: Path,
    output: Path,
    cycles: int,
    quick: bool,
) -> dict[str, object]:
    """Locally attack every active UAV block with a full three-missile objective."""
    source = json.loads(plan_file.read_text(encoding="utf-8"))
    active = [strategy_from_record(item) for item in source["bombs"]]
    search_points = q4.surface_points(20 if quick else 32, 5 if quick else 9, 4 if quick else 6)
    acceptance_points = q4.surface_points(64, 15, 10)
    history: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    bounds_by_uav = {
        uav: BOUNDS_ROUTE + [(0.0, 1.0)] * (2 * sum(item.uav == uav for item in active))
        for uav in UAVS
    }
    for cycle in range(1, cycles + 1):
        changed = False
        for uav in UAVS:
            current_group = [item for item in active if item.uav == uav]
            fixed = [item for item in active if item.uav != uav]
            warm, labels = encode_mixed_block(current_group)
            before = metrics(active, 0.02, acceptance_points)
            result = minimize(
                lambda x: -float(
                    metrics(
                        fixed + decode_mixed_block(uav, labels, x),
                        0.10 if quick else 0.06,
                        search_points,
                    )["total"]
                ),
                warm,
                method="Nelder-Mead",
                bounds=bounds_by_uav[uav],
                options={
                    "maxiter": 120 if quick else 280,
                    "xatol": 2e-7,
                    "fatol": 2e-7,
                },
            )
            candidate = fixed + decode_mixed_block(uav, labels, result.x)
            after = metrics(candidate, 0.02, acceptance_points)
            improvement = float(after["total"]) - float(before["total"])
            accepted = improvement > 0.005
            if accepted:
                active = candidate
                changed = True
            history.append(
                {
                    "cycle": cycle,
                    "uav": uav,
                    "accepted": accepted,
                    "before_total": before["total"],
                    "candidate_total": after["total"],
                    "improvement": improvement,
                    "optimizer_success": bool(result.success),
                    "optimizer_evaluations": int(result.nfev),
                }
            )
            print(
                f"block_cycle={cycle},uav={uav},accepted={accepted},"
                f"improvement={improvement:.9f},evaluations={result.nfev}",
                flush=True,
            )
        if not changed:
            break

    renumbered = []
    for uav in UAVS:
        group = sorted(
            (item for item in active if item.uav == uav),
            key=lambda item: item.release_time,
        )
        for number, item in enumerate(group, 1):
            renumbered.append(
                BombStrategy(
                    item.uav,
                    number,
                    item.theta,
                    item.speed,
                    item.release_time,
                    item.delay,
                    item.assigned_missile,
                    item.gravity,
                )
            )
    final = metrics(renumbered, 0.01, q4.surface_points(80, 19, 12))
    record = {
        "source": str(plan_file.resolve()),
        "threshold": 0.005,
        "history": history,
        "metrics": final,
        "bombs": [strategy_record(item) for item in renumbered],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def multistart_active_blocks(
    plan_file: Path,
    output: Path,
    seeds: list[int],
    quick: bool,
) -> dict[str, object]:
    """Attack the nontrivial mixed blocks from independent perturbed starts."""
    source = json.loads(plan_file.read_text(encoding="utf-8"))
    active = [strategy_from_record(item) for item in source["bombs"]]
    search_points = q4.surface_points(20 if quick else 32, 5 if quick else 9, 4 if quick else 6)
    acceptance_points = q4.surface_points(64, 15, 10)
    history: list[dict[str, object]] = []
    for uav in ("FY3", "FY4", "FY5"):
        current_group = [item for item in active if item.uav == uav]
        fixed = [item for item in active if item.uav != uav]
        warm, labels = encode_mixed_block(current_group)
        count = len(current_group)
        bounds = BOUNDS_ROUTE + [(0.0, 1.0)] * (2 * count)
        lower = np.array([a for a, _ in bounds])
        upper = np.array([b for _, b in bounds])
        scale = np.array([math.radians(4.0), 5.0, *([0.035] * (2 * count))])
        before = metrics(active, 0.02, acceptance_points)
        proposals = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 1009 * list(UAVS).index(uav))
            start = np.clip(warm + rng.normal(size=len(warm)) * scale, lower, upper)
            result = minimize(
                lambda x: -float(
                    metrics(
                        fixed + decode_mixed_block(uav, labels, x),
                        0.10 if quick else 0.06,
                        search_points,
                    )["total"]
                ),
                start,
                method="Nelder-Mead",
                bounds=bounds,
                options={
                    "maxiter": 160 if quick else 360,
                    "xatol": 2e-7,
                    "fatol": 2e-7,
                },
            )
            candidate = fixed + decode_mixed_block(uav, labels, result.x)
            audited = metrics(candidate, 0.02, acceptance_points)
            proposals.append((float(audited["total"]), candidate, audited, seed, result))
            attempts.append(
                {
                    "uav": uav,
                    "seed": seed,
                    "candidate_total": audited["total"],
                    "optimizer_success": bool(result.success),
                    "optimizer_evaluations": int(result.nfev),
                }
            )
            print(
                f"multistart_uav={uav},seed={seed},candidate={audited['total']:.9f},"
                f"evaluations={result.nfev}",
                flush=True,
            )
        total, candidate, audited, seed, result = max(proposals, key=lambda item: item[0])
        improvement = total - float(before["total"])
        accepted = improvement > 0.005
        if accepted:
            active = candidate
        history.append(
            {
                "uav": uav,
                "accepted": accepted,
                "seed": seed,
                "before_total": before["total"],
                "candidate_total": audited["total"],
                "improvement": improvement,
                "optimizer_success": bool(result.success),
                "optimizer_evaluations": int(result.nfev),
            }
        )
        print(
            f"multistart_accept_uav={uav},accepted={accepted},improvement={improvement:.9f}",
            flush=True,
        )

    renumbered = []
    for uav in UAVS:
        group = sorted(
            (item for item in active if item.uav == uav),
            key=lambda item: item.release_time,
        )
        for number, item in enumerate(group, 1):
            renumbered.append(
                BombStrategy(
                    item.uav,
                    number,
                    item.theta,
                    item.speed,
                    item.release_time,
                    item.delay,
                    item.assigned_missile,
                    item.gravity,
                )
            )
    final = metrics(renumbered, 0.01, q4.surface_points(80, 19, 12))
    record = {
        "source": str(plan_file.resolve()),
        "seeds": seeds,
        "threshold": 0.005,
        "attempts": attempts,
        "history": history,
        "metrics": final,
        "bombs": [strategy_record(item) for item in renumbered],
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def export_bomb_table(plan_file: Path, output: Path) -> list[dict[str, object]]:
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    strategies = [strategy_from_record(item) for item in plan["bombs"]]
    points = q4.surface_points(96, 21, 14)
    rows = []
    for item in strategies:
        duration, intervals, minimum = single_metrics(
            item, item.assigned_missile, 0.005, points
        )
        record = strategy_record(item)
        record.update(
            {
                "individual_duration": duration,
                "individual_intervals": intervals,
                "minimum_excess": minimum,
            }
        )
        rows.append(record)
        print(
            f"bomb={item.uav}-{item.number},missile={item.assigned_missile},"
            f"duration={duration:.9f}",
            flush=True,
        )
    output.write_text(json.dumps({"bombs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def compatibility_matrix(seed: int, quick: bool) -> dict[str, dict[str, BombStrategy]]:
    return {
        uav: {
            missile: optimize_single(uav, missile, seed + 101 * i + 17 * k, quick)
            for k, missile in enumerate(MISSILES)
        }
        for i, uav in enumerate(UAVS)
    }


def q4_m1_baseline() -> list[BombStrategy]:
    return [
        BombStrategy(
            strategy.name,
            1,
            strategy.theta,
            strategy.speed,
            strategy.release_time,
            strategy.delay,
            "M1",
        )
        for strategy in q4.final_total_candidate()
    ]


def self_test() -> None:
    baseline = q4_m1_baseline()
    points = q4.surface_points(24, 5, 4)
    result = metrics(baseline, 0.05, points)
    expected = q4.metrics(q4.final_total_candidate(), 0.05, points)[1]
    assert abs(float(result["durations"]["M1"]) - expected) < 1e-10
    assert float(result["durations"]["M2"]) >= 0.0
    assert missile_hit_time("M3") < missile_hit_time("M2") < missile_hit_time("M1")
    triad = decode_triad("FY2", "M2", np.array([1.0, 100.0, 0.2, 0.4, 0.6, 0.3, 0.5, 0.7]))
    validate(triad)
    assert all(
        right.release_time - left.release_time >= 1.0 - 1e-10
        for left, right in zip(triad, triad[1:])
    )
    known = warm_triad(decode_one("FY1", "M1", np.array([0.0, 140.0, 1.0, 0.0])), "M1")
    recovered = decode_triad("FY1", "M1", encode_triad(known, "M1"))
    assert np.allclose(
        [item.release_time for item in known],
        [item.release_time for item in recovered],
        atol=1e-9,
    )
    mixed = [
        BombStrategy("FY4", 1, 1.2, 100.0, 3.0, 2.0, "M2"),
        BombStrategy("FY4", 2, 1.2, 100.0, 9.0, 3.0, "M3"),
    ]
    encoded_mixed, mixed_labels = encode_mixed_block(mixed)
    recovered_mixed = decode_mixed_block("FY4", mixed_labels, encoded_mixed)
    assert np.allclose(
        [item.release_time for item in mixed],
        [item.release_time for item in recovered_mixed],
        atol=1e-9,
    )
    assert np.allclose(
        [item.delay for item in mixed],
        [item.delay for item in recovered_mixed],
        atol=1e-9,
    )
    assert intersect_intervals([(0.0, 2.0), (3.0, 5.0)], [(1.0, 4.0)]) == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compatibility", action="store_true")
    parser.add_argument("--triad", help="optimize one UAV,missile pair, for example FY1,M1")
    parser.add_argument("--triads", action="store_true", help="build all 15 triad candidates")
    parser.add_argument(
        "--enumerate-combinations",
        action="store_true",
        help="rank all 3^5 combinations from the saved triad library",
    )
    parser.add_argument(
        "--audit-combinations",
        action="store_true",
        help="recompute the screened leaders on a denser grid",
    )
    parser.add_argument(
        "--refine-combination",
        action="store_true",
        help="run one block-coordinate pass from an audited leader",
    )
    parser.add_argument(
        "--recombine-refined",
        action="store_true",
        help="enumerate UAV-block recombinations of saved refined plans",
    )
    parser.add_argument(
        "--validate-balanced",
        action="store_true",
        help="run convergence, marginal, rounding, and gravity checks",
    )
    parser.add_argument(
        "--activate-unused",
        action="store_true",
        help="reuse zero-marginal bomb slots on their fixed UAV routes",
    )
    parser.add_argument(
        "--validate-activated",
        action="store_true",
        help="validate the activated mixed-missile plan",
    )
    parser.add_argument(
        "--export-bomb-table",
        action="store_true",
        help="compute per-bomb durations for result3.xlsx",
    )
    parser.add_argument(
        "--attack-empty",
        action="store_true",
        help="multi-seed attack of remaining UAV bomb slots",
    )
    parser.add_argument(
        "--prune-zero-marginal",
        action="store_true",
        help="remove bombs with negligible fixed-grid leave-one-out loss",
    )
    parser.add_argument(
        "--refine-active-blocks",
        action="store_true",
        help="jointly polish each active UAV route and its mixed-missile bombs",
    )
    parser.add_argument(
        "--multistart-active-blocks",
        action="store_true",
        help="attack FY3-FY5 active blocks from independent perturbed starts",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", default="41")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q5_triad_candidates.json"),
    )
    parser.add_argument(
        "--combination-output",
        type=Path,
        default=Path(__file__).with_name("q5_candidate_combinations.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(__file__).with_name("q5_candidate_top_audit.json"),
    )
    parser.add_argument(
        "--refinement-output",
        type=Path,
        default=Path(__file__).with_name("q5_refined_plans.json"),
    )
    parser.add_argument(
        "--recombination-output",
        type=Path,
        default=Path(__file__).with_name("q5_refined_recombinations.json"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path(__file__).with_name("q5_balanced_validation.json"),
    )
    parser.add_argument(
        "--activation-output",
        type=Path,
        default=Path(__file__).with_name("q5_activated_plan.json"),
    )
    parser.add_argument(
        "--activated-validation-output",
        type=Path,
        default=Path(__file__).with_name("q5_activated_validation.json"),
    )
    parser.add_argument(
        "--bomb-table-output",
        type=Path,
        default=Path(__file__).with_name("q5_bomb_table.json"),
    )
    parser.add_argument(
        "--attack-output",
        type=Path,
        default=Path(__file__).with_name("q5_attacked_plan.json"),
    )
    parser.add_argument(
        "--pruned-output",
        type=Path,
        default=Path(__file__).with_name("q5_pruned_plan.json"),
    )
    parser.add_argument(
        "--block-attack-output",
        type=Path,
        default=Path(__file__).with_name("q5_block_attacked_plan.json"),
    )
    parser.add_argument(
        "--multistart-output",
        type=Path,
        default=Path(__file__).with_name("q5_multistart_plan.json"),
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--prune-threshold", type=float, default=1e-6)
    parser.add_argument("--only-uav", choices=list(UAVS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    self_test()
    print("missile_hit_times", {name: missile_hit_time(name) for name in MISSILES})
    seeds = [int(item) for item in args.seeds.split(",")]
    if args.prune_zero_marginal:
        record = prune_zero_marginal_bombs(
            args.activated_validation_output,
            args.pruned_output,
            args.prune_threshold,
        )
        print(f"saved={args.pruned_output}")
        print(
            json.dumps(
                {"removed": record["removed"], "metrics": record["metrics"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.multistart_active_blocks:
        record = multistart_active_blocks(
            args.activation_output,
            args.multistart_output,
            seeds,
            args.quick,
        )
        print(f"saved={args.multistart_output}")
        print(json.dumps({"metrics": record["metrics"], "history": record["history"]}, ensure_ascii=False, indent=2))
        return
    if args.refine_active_blocks:
        record = refine_active_blocks(
            args.activation_output,
            args.block_attack_output,
            args.cycles,
            args.quick,
        )
        print(f"saved={args.block_attack_output}")
        print(json.dumps({"metrics": record["metrics"], "history": record["history"]}, ensure_ascii=False, indent=2))
        return
    if args.attack_empty:
        record = attack_empty_slots(
            args.activation_output,
            args.attack_output,
            seeds,
            args.quick,
        )
        print(f"saved={args.attack_output}")
        print(json.dumps({"metrics": record["metrics"], "history": record["history"]}, ensure_ascii=False, indent=2))
        return
    if args.export_bomb_table:
        rows = export_bomb_table(args.activation_output, args.bomb_table_output)
        print(f"saved={args.bomb_table_output},rows={len(rows)}")
        return
    if args.validate_activated:
        record = validate_balanced_plan(
            args.refinement_output,
            args.activated_validation_output,
            args.activation_output,
        )
        print(f"saved={args.activated_validation_output}")
        print(
            json.dumps(
                {
                    "dense": record["resolutions"]["dense"],
                    "rounded": record["rounded_decision_metrics"],
                    "gravity": {
                        key: value["total"] for key, value in record["gravity_sensitivity"].items()
                    },
                    "marginals": record["marginal_contributions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.activate_unused:
        record = activate_unused_bombs(args.validation_output, args.activation_output, args.quick)
        print(f"saved={args.activation_output}")
        print(json.dumps({"metrics": record["metrics"], "history": record["history"]}, ensure_ascii=False, indent=2))
        return
    if args.validate_balanced:
        record = validate_balanced_plan(args.refinement_output, args.validation_output)
        print(f"saved={args.validation_output}")
        print(
            json.dumps(
                {
                    "dense": record["resolutions"]["dense"],
                    "rounded": record["rounded_decision_metrics"],
                    "gravity": {
                        key: value["total"] for key, value in record["gravity_sensitivity"].items()
                    },
                    "marginals": record["marginal_contributions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.recombine_refined:
        ranked = recombine_refined_plans(args.refinement_output, args.recombination_output)
        print(f"saved={args.recombination_output}")
        for rank, item in enumerate(ranked, 1):
            result = item["metrics"]
            print(
                f"rank={rank},sources={item['source_plan_by_uav']},"
                f"durations={result['durations']},total={result['total']:.9f},"
                f"minimum={result['minimum']:.9f},simultaneous={result['simultaneous']:.9f}"
            )
        return
    if args.refine_combination:
        record = refine_assignment(
            args.output,
            args.audit_output,
            args.rank,
            args.refinement_output,
            args.cycles,
            args.quick,
            args.only_uav,
        )
        print(f"saved={args.refinement_output}")
        print(json.dumps({"assignment": record["assignment"], "metrics": record["metrics"]}, ensure_ascii=False, indent=2))
        return
    if args.audit_combinations:
        ranked = audit_top_combinations(
            args.output,
            args.combination_output,
            args.audit_output,
            args.top,
            0.02,
            q4.surface_points(64, 15, 10),
        )
        print(f"saved={args.audit_output}")
        for rank, item in enumerate(ranked, 1):
            result = item["metrics"]
            print(
                f"rank={rank},assignment={item['assignment']},"
                f"durations={result['durations']},total={result['total']:.9f},"
                f"minimum={result['minimum']:.9f},simultaneous={result['simultaneous']:.9f}"
            )
        return
    if args.enumerate_combinations:
        ranked = enumerate_candidate_combinations(
            args.output,
            args.combination_output,
            0.08 if args.quick else 0.03,
            q4.surface_points(32 if args.quick else 64, 7 if args.quick else 15, 5 if args.quick else 10),
        )
        print(f"saved={args.combination_output}")
        for rank, item in enumerate(ranked[:10], 1):
            result = item["metrics"]
            print(
                f"rank={rank},assignment={item['assignment']},"
                f"durations={result['durations']},total={result['total']:.9f},"
                f"minimum={result['minimum']:.9f},simultaneous={result['simultaneous']:.9f}"
            )
        return
    if args.triad:
        uav, missile = args.triad.split(",")
        if uav not in UAVS or missile not in MISSILES:
            parser.error("--triad must be one of FY1..FY5,M1..M3")
        strategies, runs = optimize_triad(uav, missile, seeds, args.quick)
        record = save_triad_candidate(args.output, uav, missile, strategies, runs, args.quick)
        print(f"saved={args.output}")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    if args.triads:
        existing: set[str] = set()
        if args.output.exists() and not args.force:
            existing = set(json.loads(args.output.read_text(encoding="utf-8")).get("candidates", {}))
        for uav in UAVS:
            for missile in MISSILES:
                key = f"{uav}-{missile}"
                if key in existing:
                    print(f"skip_existing={key}", flush=True)
                    continue
                strategies, runs = optimize_triad(uav, missile, seeds, args.quick)
                record = save_triad_candidate(
                    args.output, uav, missile, strategies, runs, args.quick
                )
                print(
                    f"saved_candidate={key},focus_duration={record['focus_duration']:.9f}",
                    flush=True,
                )
        print(f"saved={args.output}")
        return
    if not args.compatibility:
        print(metrics(q4_m1_baseline(), 0.02, q4.surface_points(48, 11, 8)))
        return
    matrix = compatibility_matrix(args.seed, args.quick)
    audit_points = q4.surface_points(64, 15, 10)
    for uav, row in matrix.items():
        for missile, strategy in row.items():
            duration, intervals, minimum = single_metrics(
                strategy, missile, 0.02, audit_points
            )
            print(
                f"{uav},{missile},duration={duration:.9f},"
                f"theta_deg={math.degrees(strategy.theta):.9f},speed={strategy.speed:.9f},"
                f"release={strategy.release_time:.9f},delay={strategy.delay:.9f},"
                f"explosion={strategy.explosion_time:.9f},minimum_excess={minimum:.9f},"
                f"intervals={intervals}"
            )


if __name__ == "__main__":
    main()
