"""CUMCM 2025 A/Q4: three-UAV joint full-surface smoke coverage.

The optimizer uses sampled geometry to find candidates.  Final candidates are
re-evaluated on a denser target-surface and time grid; no sampled score is
presented as a proof of global optimality.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

import q1_strict_occlusion as geometry
import q2_optimize as q2


G = 9.8
UAVS = {
    "FY1": np.array([17_800.0, 0.0, 1_800.0]),
    "FY2": np.array([12_000.0, 1_400.0, 1_400.0]),
    "FY3": np.array([6_000.0, -3_000.0, 700.0]),
}
BOUNDS_ONE = [(0.0, 2 * math.pi), q2.SPEED_BOUNDS, (0.0, geometry.MISSILE_HIT_TIME), (0.0, 1.0)]


@dataclass(frozen=True)
class Strategy:
    name: str
    uav0: np.ndarray
    theta: float
    speed: float
    explosion_time: float
    delay: float
    gravity: float = G

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), 0.0])

    @property
    def release_time(self) -> float:
        return self.explosion_time - self.delay

    @property
    def release_point(self) -> np.ndarray:
        return self.uav0 + self.speed * self.release_time * self.direction

    @property
    def explosion_point(self) -> np.ndarray:
        point = self.uav0 + self.speed * self.explosion_time * self.direction
        point[2] -= 0.5 * self.gravity * self.delay**2
        return point

    @property
    def active_interval(self) -> tuple[float, float]:
        return (
            self.explosion_time,
            min(self.explosion_time + geometry.SMOKE_LIFETIME, geometry.MISSILE_HIT_TIME),
        )

    def cloud_center(self, t: float) -> np.ndarray:
        return self.explosion_point - np.array(
            [0.0, 0.0, geometry.SMOKE_SINK_SPEED * (t - self.explosion_time)]
        )


def decode_one(name: str, x: np.ndarray, gravity: float = G) -> Strategy:
    theta, speed, explosion_time, delay_fraction = map(float, x)
    uav0 = UAVS[name]
    free_fall_limit = math.sqrt(2 * uav0[2] / gravity)
    delay = delay_fraction * min(explosion_time, free_fall_limit)
    return Strategy(name, uav0, theta % (2 * math.pi), speed, explosion_time, delay, gravity)


def decode(x: np.ndarray, gravity: float = G) -> list[Strategy]:
    return [decode_one(name, x[4 * i : 4 * i + 4], gravity) for i, name in enumerate(UAVS)]


def encode(strategies: list[Strategy]) -> np.ndarray:
    values: list[float] = []
    for strategy in strategies:
        limit = min(
            strategy.explosion_time,
            math.sqrt(2 * strategy.uav0[2] / strategy.gravity),
        )
        fraction = 0.0 if limit <= 0 else strategy.delay / limit
        values.extend([strategy.theta, strategy.speed, strategy.explosion_time, fraction])
    return np.array(values, dtype=float)


def verified_pair_warm() -> np.ndarray:
    """Sample-verified FY1/FY2 joint interval [12, 17.0095] plus an idle FY3."""
    strategies = [
        Strategy("FY1", UAVS["FY1"], math.pi, 140.0, 12.0, 6.158392828840035),
        decode_one("FY2", np.array([5.41666133, 139.02829241, 12.92721758, 0.30613764])),
        Strategy("FY3", UAVS["FY3"], 0.0, 70.0, 0.0, 0.0),
    ]
    return encode(strategies)


def final_continuity_candidate() -> list[Strategy]:
    """Return the dense-grid-audited continuity-first Q4 candidate."""
    return [
        Strategy(
            "FY1",
            UAVS["FY1"],
            3.14143109,
            139.99045588,
            11.77721596,
            6.074581879452,
        ),
        Strategy(
            "FY2",
            UAVS["FY2"],
            5.41668229,
            139.02365109,
            12.92501878,
            3.967951296417,
        ),
        Strategy(
            "FY3",
            UAVS["FY3"],
            math.radians(88.517916215615),
            137.413370432262,
            22.603203920401,
            4.162196318314,
        ),
    ]


def final_total_candidate() -> list[Strategy]:
    """Return the current total-duration-first Q4 candidate."""
    return [
        Strategy(
            "FY1",
            UAVS["FY1"],
            math.radians(5.185976),
            140.0,
            0.921923074,
            0.175519027,
        ),
        Strategy(
            "FY2",
            UAVS["FY2"],
            math.radians(308.105379499940),
            133.910090271799,
            12.986365112826,
            4.191458102248,
        ),
        Strategy(
            "FY3",
            UAVS["FY3"],
            math.radians(73.232775959664),
            140.0,
            23.054130015768,
            0.0,
        ),
    ]


def surface_points(n_phi: int, n_z: int, n_r: int) -> np.ndarray:
    """Sample the lateral surface and both circular end caps."""
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    side = [
        np.column_stack(
            (
                geometry.TARGET_RADIUS * cos_phi,
                geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS * sin_phi,
                np.full(n_phi, z),
            )
        )
        for z in np.linspace(0.0, geometry.TARGET_HEIGHT, n_z)
    ]
    caps: list[np.ndarray] = []
    radii = np.linspace(0.0, geometry.TARGET_RADIUS, n_r + 1)[:-1]
    for z in (0.0, geometry.TARGET_HEIGHT):
        caps.append(np.array([[0.0, geometry.TARGET_CENTER_XY[1], z]]))
        for radius in radii[1:]:
            caps.append(
                np.column_stack(
                    (
                        radius * cos_phi,
                        geometry.TARGET_CENTER_XY[1] + radius * sin_phi,
                        np.full(n_phi, z),
                    )
                )
            )
    return np.vstack(side + caps)


def joint_point_distances(
    strategies: list[Strategy], times: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Return min_i dist(C_i(t), segment[M(t), P]) for every time and point."""
    missiles = geometry.MISSILE_0 + (
        geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    )
    best = np.full((len(times), len(points)), np.inf)
    for strategy in strategies:
        start, end = strategy.active_interval
        active = (times >= start - 1e-12) & (times <= end + 1e-12)
        if not np.any(active):
            continue
        active_times = times[active]
        active_missiles = missiles[active]
        clouds = strategy.explosion_point[None, :] - np.column_stack(
            (
                np.zeros(len(active_times)),
                np.zeros(len(active_times)),
                geometry.SMOKE_SINK_SPEED * (active_times - strategy.explosion_time),
            )
        )
        segments = points[None, :, :] - active_missiles[:, None, :]
        offsets = clouds - active_missiles
        lam = np.einsum("tpi,ti->tp", segments, offsets) / np.einsum(
            "tpi,tpi->tp", segments, segments
        )
        lam = np.clip(lam, 0.0, 1.0)
        closest = active_missiles[:, None, :] + lam[:, :, None] * segments
        distances = np.linalg.norm(closest - clouds[:, None, :], axis=2)
        best[active] = np.minimum(best[active], distances)
    return best


def joint_worst_distances(
    strategies: list[Strategy], times: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Return max_P min_i dist(C_i(t), segment[M(t), P])."""
    return np.max(joint_point_distances(strategies, times, points), axis=1)


def joint_worst_distances_chunked(
    strategies: list[Strategy], times: np.ndarray, points: np.ndarray, chunk: int = 64
) -> np.ndarray:
    if len(times) * len(points) <= 2_000_000:
        return joint_worst_distances(strategies, times, points)
    return np.concatenate(
        [
            joint_worst_distances(strategies, times[index : index + chunk], points)
            for index in range(0, len(times), chunk)
        ]
    )


def sampled_intervals(times: np.ndarray, excess: np.ndarray) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    for t0, t1, f0, f1 in zip(times[:-1], times[1:], excess[:-1], excess[1:]):
        if not (math.isfinite(float(f0)) and math.isfinite(float(f1))):
            continue
        if f0 <= 0 and f1 <= 0:
            pieces.append((float(t0), float(t1)))
        elif f0 <= 0 < f1:
            root = t0 + (t1 - t0) * (-f0 / (f1 - f0))
            pieces.append((float(t0), float(root)))
        elif f1 <= 0 < f0:
            root = t0 + (t1 - t0) * (f0 / (f0 - f1))
            pieces.append((float(root), float(t1)))
    return geometry.merge(pieces, tol=1e-9)


def time_grid(strategies: list[Strategy], dt: float) -> np.ndarray:
    start = min(strategy.active_interval[0] for strategy in strategies)
    end = max(strategy.active_interval[1] for strategy in strategies)
    regular = np.arange(start, end + dt / 2, dt)
    events = np.array([value for strategy in strategies for value in strategy.active_interval])
    return np.unique(np.clip(np.r_[regular, events, end], start, end))


def metrics(
    strategies: list[Strategy], dt: float, points: np.ndarray
) -> tuple[float, float, list[tuple[float, float]], float]:
    times = time_grid(strategies, dt)
    excess = joint_worst_distances_chunked(strategies, times, points) - geometry.SMOKE_RADIUS
    intervals = sampled_intervals(times, excess)
    total = sum(b - a for a, b in intervals)
    longest = max((b - a for a, b in intervals), default=0.0)
    finite = excess[np.isfinite(excess)]
    minimum = float(np.min(finite)) if len(finite) else math.inf
    return longest, total, intervals, minimum


def score(
    x: np.ndarray,
    dt: float,
    points: np.ndarray,
    guided: bool,
    total_first: bool = False,
) -> float:
    longest, total, _, minimum = metrics(decode(x), dt, points)
    primary, secondary = (total, longest) if total_first else (longest, total)
    if not guided:
        return -(primary + 1e-4 * secondary)
    near = math.exp(-max(0.0, minimum) / 12.0) if math.isfinite(minimum) else 0.0
    return -(primary + 0.03 * secondary + 0.30 * near)


def coverage_loss(strategies: list[Strategy], times: np.ndarray, points: np.ndarray) -> float:
    distances = joint_point_distances(strategies, times, points)
    excess = np.maximum(distances - geometry.SMOKE_RADIUS, 0.0)
    finite = excess[np.isfinite(excess)]
    if len(finite) != excess.size:
        return 1e6
    return float(np.mean(excess**2) + 0.3 * np.max(excess) ** 2)


def interval_bounds(
    start: float, end: float, anchor_index: int | None = 0
) -> list[tuple[float, float]]:
    if end - start > geometry.SMOKE_LIFETIME:
        raise ValueError("all-active interval parameterization requires length <= smoke lifetime")
    bounds: list[tuple[float, float]] = []
    for index in range(len(UAVS)):
        explosion_bounds = (
            (max(0.0, end - geometry.SMOKE_LIFETIME), start)
            if index == anchor_index
            else (max(0.0, start - geometry.SMOKE_LIFETIME), end)
        )
        bounds.extend([BOUNDS_ONE[0], BOUNDS_ONE[1], explosion_bounds, BOUNDS_ONE[3]])
    return bounds


def instantaneous_candidate(
    t: float,
    seed: int,
    quick: bool,
    bounds: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    bounds = interval_bounds(t, t) if bounds is None else bounds
    points = surface_points(32 if quick else 48, 7 if quick else 11, 5 if quick else 8)
    result = differential_evolution(
        lambda x: coverage_loss(decode(x), np.array([t]), points),
        bounds,
        seed=seed,
        popsize=10 if quick else 13,
        maxiter=85 if quick else 130,
        tol=2e-6,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return result.x.copy()


def interval_population(
    seed: int, warm: np.ndarray, bounds: list[tuple[float, float]], size: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([a for a, _ in bounds])
    upper = np.array([b for _, b in bounds])
    population = rng.uniform(lower, upper, size=(size, len(bounds)))
    count = size // 2
    scales = np.tile(np.array([0.35, 9.0, 1.2, 0.12]), len(UAVS))
    population[:count] = np.clip(
        warm + rng.normal(size=(count, len(bounds))) * scales,
        lower,
        upper,
    )
    population[0] = np.clip(warm, lower, upper)
    return population


def solve_fixed_interval(
    start: float, end: float, seeds: list[int], quick: bool, anchor_index: int
) -> tuple[list[Strategy], list[tuple[int, float]]]:
    target_bounds = interval_bounds(start, end, anchor_index)
    midpoint = (start + end) / 2
    warm = instantaneous_candidate(midpoint, seeds[0] + 17, quick, target_bounds)
    if anchor_index == 0:
        known = np.clip(
            verified_pair_warm(),
            [a for a, _ in target_bounds],
            [b for _, b in target_bounds],
        )
        probe_points = surface_points(20, 5, 3)
        probe_times = np.linspace(start, end, 21)
        warm = min(
            [warm, known],
            key=lambda x: coverage_loss(decode(x), probe_times, probe_points),
        )
    coarse_points = surface_points(20 if quick else 28, 5 if quick else 7, 3 if quick else 4)
    runs: list[tuple[int, float]] = []
    half_width = (end - start) / 2
    for stage_index, fraction in enumerate((0.10, 0.20, 0.35, 0.55, 0.75, 1.00), 1):
        stage_start = midpoint - fraction * half_width
        stage_end = midpoint + fraction * half_width
        bounds = interval_bounds(stage_start, stage_end, anchor_index)
        coarse_times = np.linspace(
            stage_start,
            stage_end,
            max(5, int(math.ceil((stage_end - stage_start) / (0.16 if quick else 0.10))) + 1),
        )
        stage_candidates = [np.clip(warm, [a for a, _ in bounds], [b for _, b in bounds])]
        for seed in seeds:
            size = 72 if quick else 108
            result = differential_evolution(
                lambda x: coverage_loss(decode(x), coarse_times, coarse_points),
                bounds,
                seed=seed + 7919 * stage_index,
                init=interval_population(seed + 7919 * stage_index, warm, bounds, size),
                maxiter=45 if quick else 75,
                tol=2e-6,
                polish=False,
                workers=1,
                updating="immediate",
            )
            stage_candidates.append(result.x.copy())
            runs.append((seed + 1000 * stage_index, float(result.fun)))
        warm = min(
            stage_candidates,
            key=lambda x: coverage_loss(decode(x), coarse_times, coarse_points),
        )
    medium_points = surface_points(40 if quick else 56, 9 if quick else 13, 6 if quick else 9)
    medium_times = np.linspace(start, end, max(7, int(math.ceil((end - start) / 0.04)) + 1))
    refined = minimize(
        lambda x: coverage_loss(decode(x), medium_times, medium_points),
        warm,
        method="Nelder-Mead",
        bounds=target_bounds,
        options={"maxiter": 500 if quick else 900, "xatol": 2e-8, "fatol": 2e-10},
    )
    return decode(refined.x), runs


def one_score(x: np.ndarray, name: str, dt: float, n_ring: int) -> float:
    strategy = decode_one(name, x)
    start, end = strategy.active_interval
    if end <= start:
        return 1e4
    times = np.linspace(start, end, max(2, int(math.ceil((end - start) / dt)) + 1))
    excess = q2.sampled_worst_distances(strategy, times, n_ring) - geometry.SMOKE_RADIUS
    duration = q2.nonpositive_duration(times, excess)
    near = math.exp(-max(0.0, float(np.min(excess))) / 12.0)
    return -(duration + 0.3 * near)


def one_sampled_intervals(
    strategy: Strategy, dt: float, n_ring: int
) -> tuple[list[tuple[float, float]], float]:
    start, end = strategy.active_interval
    times = np.linspace(start, end, max(2, int(math.ceil((end - start) / dt)) + 1))
    excess = q2.sampled_worst_distances(strategy, times, n_ring) - geometry.SMOKE_RADIUS
    return sampled_intervals(times, excess), float(np.min(excess))


def handoff_score(
    x: np.ndarray, name: str, handoff: float, dt: float, n_ring: int
) -> float:
    strategy = decode_one(name, x)
    if not strategy.active_interval[0] <= handoff <= strategy.active_interval[1]:
        return 1e4 + abs(strategy.explosion_time - handoff)
    handoff_excess = float(
        q2.sampled_worst_distances(strategy, np.array([handoff]), n_ring)[0]
        - geometry.SMOKE_RADIUS
    )
    intervals, _ = one_sampled_intervals(strategy, dt, n_ring)
    if not intervals:
        return 100.0 + max(0.0, handoff_excess) ** 2
    a, b = min(intervals, key=lambda pair: max(pair[0] - handoff, handoff - pair[1], 0.0))
    gap = max(a - handoff, handoff - b, 0.0)
    extension = max(0.0, b - max(a, handoff))
    return 1000.0 * max(0.0, handoff_excess) ** 2 + 100.0 * gap**2 - extension


def exact_handoff_interval(strategy: Strategy, handoff: float) -> tuple[float, float] | None:
    intervals = one_intervals(strategy)
    containing = [(a, b) for a, b in intervals if a - 2e-6 <= handoff <= b + 2e-6]
    return max(containing, key=lambda pair: pair[1]) if containing else None


def optimize_handoff(name: str, handoff: float, seeds: list[int], quick: bool) -> Strategy | None:
    bounds = [
        BOUNDS_ONE[0],
        BOUNDS_ONE[1],
        (max(0.0, handoff - geometry.SMOKE_LIFETIME), handoff),
        BOUNDS_ONE[3],
    ]
    candidates: list[np.ndarray] = []
    for seed in seeds:
        result = differential_evolution(
            lambda x: handoff_score(x, name, handoff, 0.12 if quick else 0.07, 32 if quick else 48),
            bounds,
            seed=seed,
            popsize=8 if quick else 11,
            maxiter=45 if quick else 80,
            tol=3e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidates.append(result.x.copy())
    best = min(candidates, key=lambda x: handoff_score(x, name, handoff, 0.04, 128))
    refined = minimize(
        lambda x: handoff_score(x, name, handoff, 0.025, 192),
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 420 if quick else 750, "xatol": 2e-8, "fatol": 2e-8},
    )
    strategy = decode_one(name, refined.x)
    return strategy if exact_handoff_interval(strategy, handoff) is not None else None


def relay_chain(
    first: Strategy, order: tuple[str, str], seeds: list[int], quick: bool
) -> list[Strategy] | None:
    first_interval = max(one_intervals(first), key=lambda pair: pair[1] - pair[0])
    chain = [first]
    handoff = first_interval[1]
    for offset, name in enumerate(order):
        strategy = optimize_handoff(name, handoff, [seed + 1009 * offset for seed in seeds], quick)
        if strategy is None:
            return None
        interval = exact_handoff_interval(strategy, handoff)
        if interval is None:
            return None
        chain.append(strategy)
        handoff = interval[1]
    return chain


def interval_near(
    intervals: list[tuple[float, float]], anchor: float, tolerance: float
) -> tuple[float, float] | None:
    candidates = [(a, b) for a, b in intervals if a - tolerance <= anchor <= b + tolerance]
    return max(candidates, key=lambda pair: pair[1]) if candidates else None


def joint_handoff_score(
    x: np.ndarray,
    name: str,
    fixed: list[Strategy],
    handoff: float,
    dt: float,
    points: np.ndarray,
    guided: bool,
) -> float:
    strategy = decode_one(name, x)
    if not strategy.active_interval[0] <= handoff <= strategy.active_interval[1]:
        return 1e4 + abs(strategy.explosion_time - handoff)
    strategies = fixed + [strategy]
    longest, total, intervals, _ = metrics(strategies, dt, points)
    interval = interval_near(intervals, handoff, 1.5 * dt)
    extension = 0.0 if interval is None else max(0.0, interval[1] - handoff)
    if not guided:
        return -(extension + 1e-4 * total)

    probe_end = min(handoff + 6.0, strategy.active_interval[1])
    probe_times = np.linspace(handoff, probe_end, 25)
    probe_excess = joint_worst_distances(strategies, probe_times, points) - geometry.SMOKE_RADIUS
    finite = np.isfinite(probe_excess)
    soft = float(np.mean(np.exp(-np.maximum(0.0, probe_excess[finite]) / 6.0))) if np.any(finite) else 0.0
    return -(extension + 0.35 * soft + 1e-3 * longest)


def optimize_joint_handoff(
    name: str,
    fixed: list[Strategy],
    handoff: float,
    seeds: list[int],
    quick: bool,
) -> Strategy | None:
    bounds = [
        BOUNDS_ONE[0],
        BOUNDS_ONE[1],
        (max(0.0, handoff - geometry.SMOKE_LIFETIME), handoff),
        BOUNDS_ONE[3],
    ]
    coarse_points = surface_points(20 if quick else 28, 5 if quick else 7, 3 if quick else 4)
    candidates: list[np.ndarray] = []
    for seed in seeds:
        probe_times = np.linspace(handoff + 0.02, min(handoff + 0.6, geometry.MISSILE_HIT_TIME), 5)
        fixed_time = differential_evolution(
            lambda x: float(
                np.max(
                    joint_worst_distances(
                        fixed + [decode_one(name, x)], probe_times, coarse_points
                    )
                )
            ),
            bounds,
            seed=seed,
            popsize=7 if quick else 10,
            maxiter=35 if quick else 65,
            tol=2e-5,
            polish=True,
            workers=1,
            updating="immediate",
        )
        candidates.append(fixed_time.x.copy())
        result = differential_evolution(
            lambda x: joint_handoff_score(
                x, name, fixed, handoff, 0.18 if quick else 0.11, coarse_points, True
            ),
            bounds,
            seed=seed,
            popsize=7 if quick else 10,
            maxiter=35 if quick else 65,
            tol=4e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidates.append(result.x.copy())

    medium_points = surface_points(36 if quick else 48, 9 if quick else 11, 6 if quick else 8)
    best = min(
        candidates,
        key=lambda x: joint_handoff_score(x, name, fixed, handoff, 0.06, medium_points, False),
    )
    refined = minimize(
        lambda x: joint_handoff_score(x, name, fixed, handoff, 0.04, medium_points, False),
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 300 if quick else 600, "xatol": 2e-8, "fatol": 2e-8},
    )
    strategy = decode_one(name, refined.x)
    audit_points = surface_points(64, 15, 10)
    interval = interval_near(metrics(fixed + [strategy], 0.02, audit_points)[2], handoff, 0.03)
    return strategy if interval is not None and interval[1] > handoff + 1e-4 else None


def joint_relay_chain(
    first: Strategy, order: tuple[str, str], seeds: list[int], quick: bool
) -> list[Strategy] | None:
    first_interval = max(one_intervals(first), key=lambda pair: pair[1] - pair[0])
    chain = [first]
    handoff = first_interval[1]
    audit_points = surface_points(64, 15, 10)
    for offset, name in enumerate(order):
        strategy = optimize_joint_handoff(
            name,
            chain,
            handoff,
            [seed + 2029 * offset for seed in seeds],
            quick,
        )
        if strategy is None:
            return None
        chain.append(strategy)
        interval = interval_near(metrics(chain, 0.02, audit_points)[2], handoff, 0.03)
        if interval is None:
            return None
        handoff = interval[1]
    return chain


def optimize_one(name: str, seed: int, quick: bool) -> Strategy:
    result = differential_evolution(
        lambda x: one_score(x, name, 0.18 if quick else 0.12, 20 if quick else 28),
        BOUNDS_ONE,
        seed=seed,
        popsize=7 if quick else 9,
        maxiter=30 if quick else 55,
        tol=4e-4,
        polish=False,
        workers=1,
        updating="immediate",
    )
    refined = minimize(
        lambda x: one_score(x, name, 0.04, 96),
        result.x,
        method="Nelder-Mead",
        bounds=BOUNDS_ONE,
        options={"maxiter": 260 if quick else 520, "xatol": 2e-7, "fatol": 2e-7},
    )
    return decode_one(name, refined.x)


def refine_one_exact(strategy: Strategy, maxiter: int = 900) -> Strategy:
    """Polish one strategy against the analytic full-cylinder observer."""
    refined = minimize(
        lambda x: -q2.c_duration(decode_one(strategy.name, x), 0.02),
        encode([strategy]),
        method="Nelder-Mead",
        bounds=BOUNDS_ONE,
        options={"maxiter": maxiter, "xatol": 2e-9, "fatol": 2e-10},
    )
    return decode_one(strategy.name, refined.x)


def initial_population(seed: int, warm: np.ndarray, size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = np.array([a for _ in UAVS for a, _ in BOUNDS_ONE])
    upper = np.array([b for _ in UAVS for _, b in BOUNDS_ONE])
    population = rng.uniform(lower, upper, size=(size, len(lower)))
    scale_one = np.array([math.radians(8.0), 8.0, 3.0, 0.12])
    count = min(size // 2, 40)
    population[:count] = np.clip(
        warm + rng.normal(size=(count, len(lower))) * np.tile(scale_one, len(UAVS)),
        lower,
        upper,
    )
    population[0] = warm
    return population


def optimize_joint(
    warm: list[Strategy], seeds: list[int], quick: bool, total_first: bool
) -> tuple[list[Strategy], list[tuple[int, float, float, float]]]:
    bounds = BOUNDS_ONE * len(UAVS)
    warm_x = encode(warm)
    coarse_points = q2.ring_points(16 if quick else 24)
    candidates = [warm_x]
    runs: list[tuple[int, float, float, float]] = []
    for seed in seeds:
        population_size = 60 if quick else 96
        result = differential_evolution(
            lambda x: score(x, 0.28 if quick else 0.20, coarse_points, True, total_first),
            bounds,
            seed=seed,
            init=initial_population(seed, warm_x, population_size),
            maxiter=35 if quick else 75,
            tol=4e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        candidate = decode(result.x)
        longest, total, _, _ = metrics(candidate, 0.10, q2.ring_points(72))
        candidates.append(result.x.copy())
        runs.append((seed, -float(result.fun), longest, total))

    ring_points = q2.ring_points(72)
    best = min(candidates, key=lambda x: score(x, 0.08, ring_points, False, total_first))
    refined = minimize(
        lambda x: score(x, 0.06, ring_points, False, total_first),
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 320 if quick else 650, "xatol": 3e-7, "fatol": 3e-7},
    ).x

    # Joint coverage is not convex, so the interior of the target surface must
    # be checked after the cheap two-rim search.
    full_points = surface_points(28 if quick else 40, 6 if quick else 9, 4 if quick else 6)
    polished = minimize(
        lambda x: score(x, 0.08 if quick else 0.05, full_points, False, total_first),
        refined,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 220 if quick else 480, "xatol": 2e-7, "fatol": 2e-7},
    ).x
    return decode(polished), runs


def one_intervals(strategy: Strategy) -> list[tuple[float, float]]:
    return q2.c_intervals(strategy, 0.01)


def print_strategy(strategy: Strategy) -> None:
    intervals = one_intervals(strategy)
    print(strategy.name)
    print(f"  theta_deg={math.degrees(strategy.theta):.12f}")
    print(f"  speed={strategy.speed:.12f}")
    print(f"  release_time={strategy.release_time:.12f}")
    print(f"  release_point={strategy.release_point.tolist()}")
    print(f"  delay={strategy.delay:.12f}")
    print(f"  explosion_time={strategy.explosion_time:.12f}")
    print(f"  explosion_point={strategy.explosion_point.tolist()}")
    print(f"  individual_intervals={intervals}")
    print(f"  individual_duration={sum(b - a for a, b in intervals):.12f}")


def self_test() -> None:
    fy1 = Strategy(
        "FY1",
        UAVS["FY1"],
        math.radians(5.185976),
        140.0,
        0.921923074,
        0.175519027,
    )
    times = np.linspace(*fy1.active_interval, 41)
    points = q2.ring_points(32)
    joint = joint_worst_distances([fy1], times, points)
    single = q2.sampled_worst_distances(fy1, times, 32)
    assert np.allclose(joint, single, atol=1e-10)
    duplicate = joint_worst_distances([fy1, fy1], times, points)
    assert np.allclose(joint, duplicate, atol=1e-10)
    surface = surface_points(24, 5, 4)
    assert surface.shape[1] == 3
    assert np.all((surface[:, 2] >= 0) & (surface[:, 2] <= geometry.TARGET_HEIGHT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="41,137")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--relay-only", action="store_true")
    parser.add_argument("--joint-relay-only", action="store_true")
    parser.add_argument("--total-first", action="store_true")
    parser.add_argument("--feasibility", help="solve one continuous interval as START,END")
    parser.add_argument("--anchor", choices=list(UAVS), default="FY1")
    args = parser.parse_args()
    self_test()

    if args.final_only:
        candidate = final_total_candidate()
        print("final_total_candidate")
        individual: list[tuple[float, float]] = []
        for strategy in candidate:
            print_strategy(strategy)
            individual.extend(one_intervals(strategy))
        exact_union = geometry.merge(individual)
        exact_total = sum(b - a for a, b in exact_union)
        exact_longest = max((b - a for a, b in exact_union), default=0.0)
        print(f"individual_union_intervals={exact_union}")
        print(f"individual_union_longest={exact_longest:.12f}")
        print(f"individual_union_total={exact_total:.12f}")
        dense = metrics(candidate, 0.002, surface_points(128, 31, 18))
        print(f"dense_joint_longest={dense[0]:.12f}")
        print(f"dense_joint_total={dense[1]:.12f}")
        print(f"dense_joint_intervals={dense[2]}")
        print(f"dense_minimum_excess={dense[3]:.12f}")
        return

    if args.feasibility:
        start, end = map(float, args.feasibility.split(","))
        strategies, runs = solve_fixed_interval(
            start,
            end,
            [int(item) for item in args.seeds.split(",")],
            args.quick,
            list(UAVS).index(args.anchor),
        )
        print(f"feasibility_runs={runs}")
        for strategy in strategies:
            print_strategy(strategy)
        for label, dt, points in (
            ("medium", 0.02, surface_points(64, 15, 10)),
            ("fine", 0.008, surface_points(96, 21, 14)),
        ):
            times = np.linspace(start, end, max(7, int(math.ceil((end - start) / dt)) + 1))
            loss = coverage_loss(strategies, times, points)
            worst = float(np.max(joint_worst_distances(strategies, times, points)))
            print(f"{label}_interval_loss={loss:.12f}")
            print(f"{label}_interval_worst={worst:.12f}")
            print(f"{label}_metrics={metrics(strategies, dt, points)[:3]}")
        return

    fy1 = Strategy(
        "FY1",
        UAVS["FY1"],
        math.radians(5.185976),
        140.0,
        0.921923074,
        0.175519027,
    )
    warm = [fy1, optimize_one("FY2", 211, args.quick), optimize_one("FY3", 353, args.quick)]
    print("individual_baseline")
    for strategy in warm:
        print_strategy(strategy)
    base_points = surface_points(48, 11, 8)
    base_metrics = metrics(warm, 0.025, base_points)
    print(f"baseline_joint_longest={base_metrics[0]:.12f}")
    print(f"baseline_joint_total={base_metrics[1]:.12f}")
    print(f"baseline_joint_intervals={base_metrics[2]}")
    if args.baseline_only:
        return

    if args.joint_only:
        best, runs = optimize_joint(
            warm,
            [int(item) for item in args.seeds.split(",")],
            args.quick,
            args.total_first,
        )
        print(f"joint_runs={runs}")
        print("joint_best")
        for strategy in best:
            print_strategy(strategy)
        for label, dt, points in (
            ("medium", 0.025, surface_points(64, 15, 10)),
            ("fine", 0.01, surface_points(96, 21, 14)),
        ):
            longest, total, intervals, minimum = metrics(best, dt, points)
            print(f"{label}_joint_longest={longest:.12f}")
            print(f"{label}_joint_total={total:.12f}")
            print(f"{label}_joint_intervals={intervals}")
            print(f"{label}_minimum_excess={minimum:.12f}")
        return

    seeds = [int(item) for item in args.seeds.split(",")]
    relay_candidates: list[list[Strategy]] = []
    for order in (("FY2", "FY3"), ("FY3", "FY2")):
        candidate = relay_chain(fy1, order, seeds, args.quick)
        if candidate is not None:
            relay_candidates.append(candidate)
            relay_metrics = metrics(candidate, 0.02, surface_points(64, 15, 10))
            print(f"relay_order={('FY1',) + order}")
            print(f"relay_joint_longest={relay_metrics[0]:.12f}")
            print(f"relay_joint_total={relay_metrics[1]:.12f}")
            print(f"relay_joint_intervals={relay_metrics[2]}")
            for strategy in candidate:
                print_strategy(strategy)
    relay_best = max(
        relay_candidates,
        key=lambda item: metrics(item, 0.02, surface_points(64, 15, 10))[0],
        default=warm,
    )
    if args.relay_only:
        return

    joint_relay_candidates: list[list[Strategy]] = []
    for order in (("FY2", "FY3"), ("FY3", "FY2")):
        candidate = joint_relay_chain(fy1, order, seeds, args.quick)
        if candidate is not None:
            joint_relay_candidates.append(candidate)
            relay_metrics = metrics(candidate, 0.01, surface_points(80, 19, 12))
            print(f"joint_relay_order={('FY1',) + order}")
            print(f"joint_relay_longest={relay_metrics[0]:.12f}")
            print(f"joint_relay_total={relay_metrics[1]:.12f}")
            print(f"joint_relay_intervals={relay_metrics[2]}")
            for strategy in candidate:
                print_strategy(strategy)
    joint_relay_best = max(
        joint_relay_candidates,
        key=lambda item: metrics(item, 0.02, surface_points(64, 15, 10))[0],
        default=relay_best,
    )
    if args.joint_relay_only:
        return

    best, runs = optimize_joint(
        joint_relay_best,
        seeds,
        args.quick,
        args.total_first,
    )
    print(f"joint_runs={runs}")
    print("joint_best")
    for strategy in best:
        print_strategy(strategy)
    for label, dt, points in (
        ("medium", 0.025, surface_points(64, 15, 10)),
        ("fine", 0.01, surface_points(96, 21, 14)),
    ):
        longest, total, intervals, minimum = metrics(best, dt, points)
        print(f"{label}_joint_longest={longest:.12f}")
        print(f"{label}_joint_total={total:.12f}")
        print(f"{label}_joint_intervals={intervals}")
        print(f"{label}_minimum_excess={minimum:.12f}")

    for strategy in best:
        assert 70.0 <= strategy.speed <= 140.0
        assert 0.0 <= strategy.release_time <= strategy.explosion_time <= geometry.MISSILE_HIT_TIME
        assert strategy.explosion_point[2] >= -1e-8


if __name__ == "__main__":
    main()
