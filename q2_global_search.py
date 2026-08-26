"""Stronger Q2 search: explosion-event variables plus interval constraints.

This improves feasible solutions; it does not call a heuristic run a proof of
global optimality.  Final durations are always recomputed by observer C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.optimize import direct, minimize

import q1_strict_occlusion as geometry
import q2_optimize as q2


G = 9.8
UAV_0 = q2.UAV_0
CURRENT = q2.Strategy(0.090519290678, 139.997203740, 0.9218870760289548, 0.175952994, G)


@dataclass(frozen=True)
class Event:
    explosion_time: float
    explosion_point: np.ndarray
    gravity: float = G

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

    def controls(self) -> tuple[float, float, float, float]:
        delta_xy = self.explosion_point[:2] - UAV_0[:2]
        distance = float(np.linalg.norm(delta_xy))
        speed = distance / self.explosion_time
        theta = math.atan2(float(delta_xy[1]), float(delta_xy[0])) % (2 * math.pi)
        drop = UAV_0[2] - float(self.explosion_point[2])
        delay = math.sqrt(max(0.0, 2 * drop / self.gravity))
        return theta, speed, self.explosion_time - delay, delay

    def reachable(self, tol: float = 1e-7) -> bool:
        theta, speed, release_time, delay = self.controls()
        del theta, delay
        return (
            q2.SPEED_BOUNDS[0] - tol <= speed <= q2.SPEED_BOUNDS[1] + tol
            and release_time >= -tol
            and -tol <= self.explosion_point[2] <= UAV_0[2] + tol
        )


def event_from_strategy(strategy: q2.Strategy) -> Event:
    return Event(strategy.explosion_time, strategy.explosion_point.copy(), strategy.gravity)


def event_from_scaled(x: np.ndarray) -> Event:
    """x=(te, dx/100, dy/100, vertical_drop, a, b)."""
    return Event(
        float(x[0]),
        np.array([UAV_0[0] + 100 * x[1], 100 * x[2], UAV_0[2] - x[3]], dtype=float),
    )


def scaled_from_event(event: Event, a: float, b: float) -> np.ndarray:
    return np.array(
        [
            event.explosion_time,
            (event.explosion_point[0] - UAV_0[0]) / 100,
            event.explosion_point[1] / 100,
            UAV_0[2] - event.explosion_point[2],
            a,
            b,
        ],
        dtype=float,
    )


def kinematic_constraints(x: np.ndarray) -> np.ndarray:
    te, dx100, dy100, drop, a, b = map(float, x)
    rho = 100 * math.hypot(dx100, dy100)
    return np.array(
        [
            (rho - q2.SPEED_BOUNDS[0] * te) / 100,
            (q2.SPEED_BOUNDS[1] * te - rho) / 100,
            (0.5 * G * te**2 - drop) / 100,
            a - te,
            b - a,
            te + geometry.SMOKE_LIFETIME - b,
            geometry.MISSILE_HIT_TIME - b,
        ]
    )


def interval_constraints(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    event = event_from_scaled(x)
    a, b = float(x[4]), float(x[5])
    times = a + nodes * (b - a)
    # V has units m^2; scaling keeps SLSQP finite differences well conditioned.
    cover = np.array([-q2.c_observer(event, float(t))[0] / 100 for t in times])
    return np.r_[kinematic_constraints(x), cover]


def exchange_polish(event: Event, a: float, b: float) -> tuple[Event, float, float, int]:
    """Maximize b-a and add the worst omitted time until no violation remains."""
    x = scaled_from_event(event, a, b)
    nodes = np.linspace(0.0, 1.0, 9)
    bounds = [
        (1e-4, geometry.MISSILE_HIT_TIME),
        (-100.0, 100.0),
        (-100.0, 100.0),
        (0.0, UAV_0[2]),
        (0.0, geometry.MISSILE_HIT_TIME),
        (0.0, geometry.MISSILE_HIT_TIME),
    ]
    rounds = 0
    for rounds in range(1, 13):
        result = minimize(
            lambda y: float(y[4] - y[5]),
            x,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": lambda y: interval_constraints(y, nodes)},
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )
        if result.success or np.min(interval_constraints(result.x, nodes)) > -2e-6:
            x = result.x
        event = event_from_scaled(x)
        a, b = float(x[4]), float(x[5])
        dense_nodes = np.linspace(0.0, 1.0, 2001)
        values = np.array(
            [q2.c_observer(event, float(a + s * (b - a)))[0] for s in dense_nodes]
        )
        idx = int(np.argmax(values))
        if values[idx] <= 2e-5:
            break
        nodes = np.unique(np.r_[nodes, dense_nodes[idx]])

    # Authoritative interval endpoints come from observer C, not SLSQP tolerance.
    intervals = q2.c_intervals(event, 0.005)
    if not intervals:
        raise RuntimeError("polished event has no strictly covered interval")
    a, b = max(intervals, key=lambda pair: pair[1] - pair[0])
    return event, a, b, rounds


def direct_event(x: np.ndarray) -> Event:
    """Always-feasible box map: x=(te, theta, speed, drop_fraction)."""
    te, theta, speed, fraction = map(float, x)
    max_drop = min(UAV_0[2], 0.5 * G * te**2)
    drop = fraction * max_drop
    point = UAV_0 + speed * te * np.array([math.cos(theta), math.sin(theta), 0.0])
    point[2] -= drop
    return Event(te, point)


def direct_score(x: np.ndarray) -> float:
    event = direct_event(x)
    start, end = event.active_interval
    count = max(2, int(math.ceil((end - start) / 0.10)) + 1)
    times = np.linspace(start, end, count)
    excess = q2.sampled_worst_distances(event, times, 32) - geometry.SMOKE_RADIUS
    duration = q2.nonpositive_duration(times, excess)
    if duration <= 0:
        return 100.0 + max(0.0, float(np.min(excess)))
    return -duration - 1e-5 * max(0.0, float(-np.min(excess)))


def deterministic_candidate(maxfun: int = 30_000) -> tuple[Event, float]:
    bounds = [
        (1e-4, geometry.MISSILE_HIT_TIME),
        (0.0, 2 * math.pi),
        q2.SPEED_BOUNDS,
        (0.0, 1.0),
    ]
    result = direct(
        direct_score,
        bounds,
        maxfun=maxfun,
        maxiter=maxfun,
        locally_biased=False,
        len_tol=2e-5,
    )
    return direct_event(result.x), -float(result.fun)


def certify_relaxed_upper(duration: float = 4.589) -> dict[str, float | int]:
    """Exclude a duration using a conic relaxation over every possible start time.

    The relaxation keeps only necessary conditions: 32 points on each target
    rim, 17 times, vertical sinking, and the maximum horizontal reach 140*a.
    McCormick envelopes cover each complete start-time box.  Infeasibility of
    this easier problem therefore excludes the original problem as well.
    """
    n_phi, n_time = 32, 17
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    points = np.vstack(
        [
            np.column_stack(
                (
                    geometry.TARGET_RADIUS * np.cos(phi),
                    geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS * np.sin(phi),
                    np.full(n_phi, z),
                )
            )
            for z in q2.TARGET_Z
        ]
    )
    relative_times = np.linspace(0.0, duration, n_time)
    missile_velocity = geometry.MISSILE_SPEED * geometry.MISSILE_DIRECTION

    start = cp.Variable()
    cloud_start = cp.Variable(3)
    radius = cp.Variable(nonneg=True)
    mu = cp.Variable((n_time, len(points)))
    product = cp.Variable((n_time, len(points)))
    lower = cp.Parameter(nonneg=True)
    upper = cp.Parameter(nonneg=True)
    constraints = [
        start >= lower,
        start <= upper,
        mu >= 0,
        mu <= 1,
        cp.norm(cloud_start[:2] - UAV_0[:2], 2) <= q2.SPEED_BOUNDS[1] * start,
        product >= lower * mu,
        product >= start + upper * mu - upper,
        product <= upper * mu,
        product <= start + lower * mu - lower,
    ]
    for time_index, relative_time in enumerate(relative_times):
        cloud = cloud_start + np.array([0.0, 0.0, -geometry.SMOKE_SINK_SPEED * relative_time])
        missile_base = geometry.MISSILE_0 + missile_velocity * relative_time
        for point_index, point in enumerate(points):
            segment_point = (
                point
                + mu[time_index, point_index] * (missile_base - point)
                + missile_velocity * product[time_index, point_index]
            )
            constraints.append(cp.norm(cloud - segment_point, 2) <= radius)
    problem = cp.Problem(cp.Minimize(radius), constraints)

    def box_lower_bound(a: float, b: float) -> tuple[float, str]:
        lower.value, upper.value = a, b
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-6,
            tol_feas=1e-6,
            tol_gap_rel=1e-6,
            warm_start=True,
        )
        return float(problem.value), problem.status

    stack = [(0.0, geometry.MISSILE_HIT_TIME - duration)]
    solves = pruned = 0
    unresolved: list[tuple[float, float, float, str]] = []
    while stack:
        a, b = stack.pop()
        bound, status = box_lower_bound(a, b)
        solves += 1
        if status == "optimal" and bound > 10.00015:
            pruned += 1
        elif b - a <= 5e-6:
            unresolved.append((a, b, bound, status))
        else:
            midpoint = (a + b) / 2
            stack.extend([(a, midpoint), (midpoint, b)])
        if solves >= 20_000:
            break
    return {
        "duration": duration,
        "solves": solves,
        "pruned": pruned,
        "remaining": len(stack),
        "unresolved": len(unresolved),
    }


def describe(label: str, event: Event, a: float, b: float) -> None:
    theta, speed, release_time, delay = event.controls()
    print(label)
    print(f"  theta_deg={math.degrees(theta):.12f}")
    print(f"  speed={speed:.12f}")
    print(f"  release_time={release_time:.12f}")
    print(f"  delay={delay:.12f}")
    print(f"  explosion_time={event.explosion_time:.12f}")
    print(f"  explosion_point={event.explosion_point.tolist()}")
    print(f"  interval=[{a:.12f}, {b:.12f}]")
    print(f"  duration={b-a:.12f}")
    print(f"  reachable={event.reachable()}")


def main() -> None:
    current_event = event_from_strategy(CURRENT)
    current_interval = max(q2.c_intervals(current_event, 0.005), key=lambda p: p[1] - p[0])
    describe("current", current_event, *current_interval)

    polished = exchange_polish(current_event, *current_interval)
    describe("event_interval_polish", polished[0], polished[1], polished[2])
    print(f"  exchange_rounds={polished[3]}")

    candidate, sampled_score = deterministic_candidate()
    candidate_intervals = q2.c_intervals(candidate, 0.01)
    print(f"direct_sampled_score={sampled_score:.12f}")
    if candidate_intervals:
        interval = max(candidate_intervals, key=lambda p: p[1] - p[0])
        describe("direct_before_polish", candidate, *interval)
        refined = exchange_polish(candidate, *interval)
        describe("direct_after_polish", refined[0], refined[1], refined[2])
        print(f"  exchange_rounds={refined[3]}")
    else:
        print("direct_candidate_has_no_strict_interval")

    upper = certify_relaxed_upper()
    print(f"relaxed_upper_certificate={upper}")
    assert upper["remaining"] == 0
    assert upper["unresolved"] == 0

    # Small runnable checks required for the event reparameterization.
    assert current_event.reachable()
    recovered = current_event.controls()
    assert abs(recovered[1] - CURRENT.speed) < 1e-8
    assert abs(recovered[2] - CURRENT.release_time) < 1e-8


if __name__ == "__main__":
    main()
