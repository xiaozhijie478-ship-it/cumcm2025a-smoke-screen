"""Search Q3 open upper-bound boxes for one genuinely feasible strategy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import q1_strict_occlusion as geometry
import q2_optimize as q2
import q3_optimize as q3
import q4_optimize as q4


def bounds_from_artifact(data: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    worst = data["worst_box"]
    assert isinstance(worst, dict)
    box_lo = np.asarray(worst["lo"], dtype=float)
    box_hi = np.asarray(worst["hi"], dtype=float)
    theta = data["theta_range_deg"]
    assert isinstance(theta, list)
    lo = np.r_[math.radians(theta[0]), q2.SPEED_BOUNDS[0], box_lo[2:]]
    hi = np.r_[math.radians(theta[1]), q2.SPEED_BOUNDS[1], box_hi[2:]]
    return lo, hi


def violations(x: np.ndarray, data: dict[str, object]) -> np.ndarray:
    """Return normalized positive violations of the exact physical controls."""
    theta, speed = x[:2]
    explosions, delays = x[2:5], x[5:8]
    releases = explosions - delays
    worst = data["worst_box"]
    assert isinstance(worst, dict)
    box_lo = np.asarray(worst["lo"], dtype=float)
    box_hi = np.asarray(worst["hi"], dtype=float)
    ux, uy = speed * math.cos(theta), speed * math.sin(theta)
    values = [
        max(0.0, box_lo[0] - ux, ux - box_hi[0]) / 70.0,
        max(0.0, box_lo[1] - uy, uy - box_hi[1]) / 70.0,
        *(np.maximum(0.0, -releases) / q3.TIME_LIMIT),
        max(0.0, 1.0 - (releases[1] - releases[0])) / q3.TIME_LIMIT,
        max(0.0, 1.0 - (releases[2] - releases[1])) / q3.TIME_LIMIT,
    ]
    order = [int(index) - 1 for index in data.get("explosion_order", [])]
    values.extend(
        max(0.0, explosions[left] - explosions[right]) / q3.TIME_LIMIT
        for left, right in zip(order, order[1:])
    )
    values.extend(
        max(0.0, -q2.Strategy(theta, speed, explosions[i], delays[i], q3.G).explosion_point[2])
        / q2.UAV_0[2]
        for i in range(3)
    )
    return np.asarray(values, dtype=float)


def strategies(x: np.ndarray) -> list[q2.Strategy]:
    theta, speed = map(float, x[:2])
    return [
        q2.Strategy(theta, speed, float(x[2 + i]), float(x[5 + i]), q3.G)
        for i in range(3)
    ]


def joint_components(
    x: np.ndarray,
    dt: float,
    points: np.ndarray,
) -> tuple[float, float]:
    current = strategies(x)
    times = q4.time_grid(current, dt)
    excess = (
        q4.joint_worst_distances_chunked(current, times, points)
        - geometry.SMOKE_RADIUS
    )
    intervals = q4.sampled_intervals(times, excess)
    duration = q3.union_duration(intervals)
    outside = np.isfinite(excess) & (excess > 0.0)
    near_values = np.zeros_like(times)
    near_values[outside] = np.exp(-excess[outside] / 2.5)
    near = float(np.trapezoid(near_values, times))
    return duration, near


def sample_population(
    data: dict[str, object],
    lo: np.ndarray,
    hi: np.ndarray,
    seed: int,
    size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feasible: list[np.ndarray] = []
    best: list[tuple[float, np.ndarray]] = []
    for _ in range(80):
        batch = rng.uniform(lo, hi, size=(2048, 8))
        for candidate in batch:
            error = float(np.sum(violations(candidate, data)))
            if error <= 1e-12:
                feasible.append(candidate)
            else:
                best.append((error, candidate))
        if len(best) > 4 * size:
            best = sorted(best, key=lambda item: item[0])[:size]
        if len(feasible) >= size:
            break
    if len(feasible) < 5:
        best.sort(key=lambda item: item[0])
        feasible.extend(candidate for _, candidate in best[: size - len(feasible)])
    if len(feasible) < 5:
        raise RuntimeError("could not construct a usable population")
    population = np.asarray(feasible[:size], dtype=float)
    if len(population) < size:
        copies = rng.choice(len(population), size=size - len(population), replace=True)
        population = np.vstack((population, population[copies]))
    return population


def attack(
    path: Path,
    seeds: list[int],
    quick: bool,
) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lo, hi = bounds_from_artifact(data)
    coarse_points = q4.surface_points(18 if quick else 28, 5 if quick else 7, 3 if quick else 4)
    coarse_dt = 0.08 if quick else 0.05
    population_size = 40 if quick else 72
    runs = []
    candidates = []

    def objective(x: np.ndarray) -> float:
        error = float(np.sum(violations(x, data)))
        if error > 1e-12:
            return 100.0 + 1_000.0 * error
        duration, near = joint_components(x, coarse_dt, coarse_points)
        return -(duration + 0.01 * near)

    for seed in seeds:
        result = differential_evolution(
            objective,
            list(zip(lo, hi)),
            seed=seed,
            init=sample_population(data, lo, hi, seed, population_size),
            maxiter=35 if quick else 90,
            tol=3e-4,
            polish=False,
            workers=1,
            updating="immediate",
        )
        error = float(np.max(violations(result.x, data)))
        coarse = joint_components(result.x, 0.025, q4.surface_points(48, 11, 7))[0]
        runs.append(
            {
                "seed": seed,
                "guided_objective": -float(result.fun),
                "coarse_joint_seconds": coarse,
                "max_constraint_violation": error,
            }
        )
        if error <= 1e-10:
            candidates.append(result.x.copy())

    if not candidates:
        return {
            "artifact": path.name,
            "box_upper_seconds": data["global_upper"],
            "feasible_candidate_found": False,
            "runs": runs,
        }

    audit_points = q4.surface_points(80, 19, 12)
    best = max(candidates, key=lambda x: joint_components(x, 0.01, audit_points)[0])
    audit_duration, _ = joint_components(best, 0.005, audit_points)
    current = strategies(best)
    individual_intervals = [
        pair for strategy in current for pair in q2.c_intervals(strategy, 0.01)
    ]
    return {
        "artifact": path.name,
        "box_upper_seconds": data["global_upper"],
        "feasible_candidate_found": True,
        "sampled_joint_seconds": audit_duration,
        "analytic_single_ball_union_seconds": q3.union_duration(individual_intervals),
        "max_constraint_violation": float(np.max(violations(best, data))),
        "x_theta_speed_explosions_delays": best.tolist(),
        "strategy": [
            {
                "theta_deg": math.degrees(strategy.theta),
                "speed": strategy.speed,
                "release_time": strategy.release_time,
                "explosion_time": strategy.explosion_time,
                "delay": strategy.delay,
            }
            for strategy in current
        ],
        "runs": runs,
    }


def self_test() -> None:
    incumbent = q3.INCUMBENT
    x = np.array(
        [
            incumbent[0].theta,
            incumbent[0].speed,
            *(item.explosion_time for item in incumbent),
            *(item.delay for item in incumbent),
        ]
    )
    ux = incumbent[0].speed * math.cos(incumbent[0].theta)
    uy = incumbent[0].speed * math.sin(incumbent[0].theta)
    box_values = [ux, uy, *x[2:]]
    data = {
        "worst_box": {"lo": box_values, "hi": box_values},
        "theta_range_deg": [179.0, 180.0],
        "required_active_bombs": [1, 2, 3],
        "explosion_order": [1, 2, 3],
    }
    assert np.max(violations(x, data)) <= 1e-12
    assert joint_components(x, 0.01, q4.surface_points(32, 7, 4))[0] > 7.6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--seeds", default="20260829,20260830")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("q3_worst_box_attack.json"))
    args = parser.parse_args()
    self_test()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    results = [attack(Path(path), seeds, args.quick) for path in args.artifacts]
    payload = {"seeds": seeds, "quick": args.quick, "results": results}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
