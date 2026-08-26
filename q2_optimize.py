"""CUMCM 2025 A/Q2: optimize FY1's single smoke bomb under strict occlusion.

The fast sampled observer is used only to find candidates.  Reported results
come from the exact analytic observer C and are checked by certified sampler B.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

import q1_strict_occlusion as geometry


UAV_0 = np.array([17_800.0, 0.0, 1_800.0])
SPEED_BOUNDS = (70.0, 140.0)
TARGET_Z = (0.0, geometry.TARGET_HEIGHT)
TEMPORAL_LIPSCHITZ = geometry.MISSILE_SPEED + geometry.SMOKE_SINK_SPEED


@dataclass(frozen=True)
class Strategy:
    theta: float
    speed: float
    explosion_time: float
    delay: float
    gravity: float

    @property
    def direction(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta), 0.0])

    @property
    def release_time(self) -> float:
        return self.explosion_time - self.delay

    @property
    def release_point(self) -> np.ndarray:
        return UAV_0 + self.speed * self.release_time * self.direction

    @property
    def explosion_point(self) -> np.ndarray:
        point = UAV_0 + self.speed * self.explosion_time * self.direction
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


def decode(x: np.ndarray, gravity: float) -> Strategy:
    """Map a box-constrained vector to four always-feasible physical variables."""
    theta, speed, explosion_time, delay_fraction = map(float, x)
    free_fall_limit = math.sqrt(2 * UAV_0[2] / gravity)
    delay = delay_fraction * min(explosion_time, free_fall_limit)
    return Strategy(theta % (2 * math.pi), speed, explosion_time, delay, gravity)


def ring_points(n: int) -> np.ndarray:
    phi = 2 * math.pi * np.arange(n) / n
    x = geometry.TARGET_RADIUS * np.cos(phi)
    y = geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS * np.sin(phi)
    rings = [np.column_stack((x, y, np.full(n, z))) for z in TARGET_Z]
    return np.vstack(rings)


def sampled_worst_distances(strategy: Strategy, times: np.ndarray, n_ring: int) -> np.ndarray:
    """Fast, non-certified observer used only inside the optimizer."""
    points = ring_points(n_ring)
    missiles = geometry.MISSILE_0 + (
        geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    )
    clouds = strategy.explosion_point[None, :] - np.column_stack(
        (np.zeros(len(times)), np.zeros(len(times)), geometry.SMOKE_SINK_SPEED * (times - strategy.explosion_time))
    )
    segments = points[None, :, :] - missiles[:, None, :]
    offsets = clouds - missiles
    lam = np.einsum("tpi,ti->tp", segments, offsets) / np.einsum(
        "tpi,tpi->tp", segments, segments
    )
    lam = np.clip(lam, 0.0, 1.0)
    closest = missiles[:, None, :] + lam[:, :, None] * segments
    distances = np.linalg.norm(closest - clouds[:, None, :], axis=2)
    return np.max(distances, axis=1)


def nonpositive_duration(times: np.ndarray, values: np.ndarray) -> float:
    """Measure {t: values(t)<=0} using linear interpolation on each time cell."""
    duration = 0.0
    for t0, t1, f0, f1 in zip(times[:-1], times[1:], values[:-1], values[1:]):
        width = float(t1 - t0)
        if f0 <= 0 and f1 <= 0:
            duration += width
        elif f0 <= 0 < f1:
            duration += width * float(-f0 / (f1 - f0))
        elif f1 <= 0 < f0:
            duration += width * float(-f1 / (f0 - f1))
    return duration


def quick_score(x: np.ndarray, gravity: float, dt: float, n_ring: int) -> float:
    strategy = decode(x, gravity)
    start, end = strategy.active_interval
    if end <= start:
        return 10_000.0 + start - end
    count = max(2, int(math.ceil((end - start) / dt)) + 1)
    times = np.linspace(start, end, count)
    excess = sampled_worst_distances(strategy, times, n_ring) - geometry.SMOKE_RADIUS
    duration = nonpositive_duration(times, excess)
    if duration <= 0:
        # Remove the all-zero plateau before a feasible screen is discovered.
        return 100.0 + max(0.0, float(np.min(excess)))
    return -duration - 1e-5 * max(0.0, float(-np.min(excess)))


def c_observer(strategy: Strategy, t: float) -> tuple[float, np.ndarray, float]:
    missile = geometry.missile_position(t)
    cloud = strategy.cloud_center(t)
    candidates = [geometry.ring_max_violation(missile, cloud, z) for z in TARGET_Z]
    vmax, witness = max(candidates, key=lambda item: item[0])
    return vmax, witness, geometry.segment_distance(missile, witness, cloud)


def bisect_boundary(strategy: Strategy, a: float, b: float, iterations: int = 60) -> float:
    fa = c_observer(strategy, a)[0]
    fb = c_observer(strategy, b)[0]
    if fa * fb > 0:
        raise ValueError("boundary is not bracketed")
    for _ in range(iterations):
        mid = (a + b) / 2
        fm = c_observer(strategy, mid)[0]
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2


def c_intervals(strategy: Strategy, step: float = 0.03) -> list[tuple[float, float]]:
    start, end = strategy.active_interval
    if end <= start:
        return []
    count = max(2, int(math.ceil((end - start) / step)) + 1)
    times = np.linspace(start, end, count)
    values = np.array([c_observer(strategy, float(t))[0] for t in times])
    roots: list[float] = []
    for idx in range(len(times) - 1):
        if values[idx] == 0:
            roots.append(float(times[idx]))
        elif values[idx] * values[idx + 1] < 0:
            roots.append(bisect_boundary(strategy, float(times[idx]), float(times[idx + 1])))
    cuts = [start] + roots + [end]
    intervals = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a and c_observer(strategy, (a + b) / 2)[0] <= 0:
            intervals.append((a, b))
    return geometry.merge(intervals)


def c_duration(strategy: Strategy, step: float = 0.03) -> float:
    return sum(b - a for a, b in c_intervals(strategy, step))


def b_bounds(strategy: Strategy, t: float, n: int) -> tuple[float, float]:
    points = ring_points(n)
    missile, cloud = geometry.missile_position(t), strategy.cloud_center(t)
    segments = points - missile
    lam = np.clip(
        ((cloud - missile) @ segments.T) / np.einsum("ij,ij->i", segments, segments),
        0.0,
        1.0,
    )
    closest = missile + lam[:, None] * segments
    lower = float(np.max(np.linalg.norm(closest - cloud, axis=1)))
    angular_gap = 2 * geometry.TARGET_RADIUS * math.sin(math.pi / (2 * n))
    return lower, lower + angular_gap


def certify_time(
    strategy: Strategy, eps_t: float = 2e-5, n_ring: int = 16_384
) -> geometry.TimeCertificate:
    start, end = strategy.active_interval
    stack = [(start, end)] if end > start else []
    covered: list[tuple[float, float]] = []
    unresolved: list[tuple[float, float]] = []
    while stack:
        a, b = stack.pop()
        mid, half = (a + b) / 2, (b - a) / 2
        lower, upper = b_bounds(strategy, mid, n_ring)
        if upper + TEMPORAL_LIPSCHITZ * half <= geometry.SMOKE_RADIUS:
            covered.append((a, b))
        elif lower - TEMPORAL_LIPSCHITZ * half > geometry.SMOKE_RADIUS:
            continue
        elif b - a <= eps_t:
            unresolved.append((a, b))
        else:
            stack.extend([(a, mid), (mid, b)])
    return geometry.TimeCertificate(geometry.merge(covered), geometry.merge(unresolved))


def optimize(gravity: float, seeds: list[int]) -> tuple[Strategy, list[tuple[int, float, np.ndarray]]]:
    bounds = [
        (0.0, 2 * math.pi),
        SPEED_BOUNDS,
        (0.0, geometry.MISSILE_HIT_TIME),
        (0.0, 1.0),
    ]
    coarse: list[tuple[int, float, np.ndarray]] = []
    for seed in seeds:
        result = differential_evolution(
            lambda x: quick_score(x, gravity, 0.12, 24),
            bounds,
            seed=seed,
            popsize=10,
            maxiter=55,
            tol=2e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        coarse.append((seed, -float(result.fun), result.x.copy()))

    fine: list[tuple[float, np.ndarray]] = []
    for _, _, x0 in sorted(coarse, key=lambda item: item[1], reverse=True)[:3]:
        result = minimize(
            lambda x: quick_score(x, gravity, 0.025, 128),
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 700, "xatol": 2e-7, "fatol": 2e-7},
        )
        fine.append((-float(result.fun), result.x.copy()))
    x_best = max(fine, key=lambda item: item[0])[1]

    # C is slower but exact in target geometry; use it for the final local polish.
    result = minimize(
        lambda x: -c_duration(decode(x, gravity), 0.05),
        x_best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 220, "xatol": 2e-8, "fatol": 2e-8},
    )
    return decode(result.x, gravity), coarse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", type=float, default=9.8)
    parser.add_argument("--seeds", default="41,137,809")
    parser.add_argument("--quick", action="store_true", help="skip the expensive B certificate")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]

    strategy, coarse = optimize(args.g, seeds)
    intervals = c_intervals(strategy, 0.01)
    duration = sum(b - a for a, b in intervals)

    print("coarse_search=")
    for seed, score, x in coarse:
        candidate = decode(x, args.g)
        print(
            f"  seed={seed}, sampled_score={score:.9f}, theta_deg={math.degrees(candidate.theta):.9f}, "
            f"v={candidate.speed:.9f}, te={candidate.explosion_time:.9f}, delay={candidate.delay:.9f}"
        )
    print(f"gravity={args.g:.9f}")
    print(f"theta_rad={strategy.theta:.12f}")
    print(f"theta_deg={math.degrees(strategy.theta):.9f}")
    print(f"direction={strategy.direction.tolist()}")
    print(f"speed={strategy.speed:.9f}")
    print(f"release_time={strategy.release_time:.9f}")
    print(f"release_point={strategy.release_point.tolist()}")
    print(f"delay={strategy.delay:.9f}")
    print(f"explosion_time={strategy.explosion_time:.9f}")
    print(f"explosion_point={strategy.explosion_point.tolist()}")
    print(f"c_intervals={intervals}")
    print(f"c_duration={duration:.9f}")
    print("c_step_stability=" + str({step: c_duration(strategy, step) for step in (0.05, 0.02, 0.01)}))

    if not args.quick:
        certificate = certify_time(strategy)
        contradictions = 0
        start, end = strategy.active_interval
        for t in np.linspace(start, end, 301):
            c_pass = c_observer(strategy, float(t))[0] <= 0
            lower, upper = b_bounds(strategy, float(t), 8192)
            if (upper <= geometry.SMOKE_RADIUS and not c_pass) or (
                lower > geometry.SMOKE_RADIUS and c_pass
            ):
                contradictions += 1
        print(f"certified_covered={certificate.covered}")
        print(f"certified_unresolved={certificate.unresolved}")
        print(
            f"certified_duration=[{certificate.lower_duration:.9f}, "
            f"{certificate.upper_duration:.9f}]"
        )
        print(f"C_vs_B_contradictions={contradictions}")
        assert contradictions == 0
        assert certificate.lower_duration <= duration <= certificate.upper_duration


if __name__ == "__main__":
    main()
