"""Safe outer branch bound for the complete Q3 control domain.

Each parameter box is allowed to choose a different strategy in every time
bin and for every target witness.  That is deliberately easier than the real
problem, so the counted duration is an upper bound for every strategy in the
box.  Splitting boxes only tightens this relaxation.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import q1_strict_occlusion as geometry
import q2_optimize as q2
import q3_optimize as q3


T = geometry.MISSILE_HIT_TIME
TAU_MAX = q3.FREE_FALL_LIMIT
TIME_RATE = geometry.MISSILE_SPEED + geometry.SMOKE_SINK_SPEED
DISTANCE_GUARD = 1e-8
TIME_GUARD = 1e-12
# FY1 can only move west at 140 m/s while M1 recedes west faster.  Beyond this
# time even the radius-10 cloud cannot touch any finite missile-target segment.
MISSILE_VX = geometry.MISSILE_SPEED * geometry.MISSILE_DIRECTION[0]
COVER_END = math.nextafter(
    (q2.UAV_0[0] - geometry.MISSILE_0[0] - geometry.SMOKE_RADIUS)
    / (q2.SPEED_BOUNDS[1] + MISSILE_VX),
    math.inf,
)


@dataclass(frozen=True)
class Box:
    """(theta, speed, r1, r2, r3, tau1, tau2, tau3) interval box."""

    lo: tuple[float, ...]
    hi: tuple[float, ...]
    depth: int = 0

    def split(self, index: int) -> tuple["Box", "Box"]:
        midpoint = (self.lo[index] + self.hi[index]) / 2
        left_hi, right_lo = list(self.hi), list(self.lo)
        left_hi[index] = midpoint
        right_lo[index] = midpoint
        return (
            Box(self.lo, tuple(left_hi), self.depth + 1),
            Box(tuple(right_lo), self.hi, self.depth + 1),
        )


def root_box(preset: str = "full") -> Box:
    if preset == "immediate":
        return Box(
            (0.0, 140.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
            (2 * math.pi, 140.0, 0.0, T - 1.0, T, 0.0, TAU_MAX, TAU_MAX),
        )
    return Box(
        (0.0, 70.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
        (2 * math.pi, 140.0, T - 2.0, T - 1.0, T, TAU_MAX, TAU_MAX, TAU_MAX),
    )


def point_box() -> Box:
    strategies = q3.INCUMBENT
    values = (
        strategies[0].theta,
        strategies[0].speed,
        *(item.release_time for item in strategies),
        *(item.delay for item in strategies),
    )
    return Box(tuple(values), tuple(values))


def contract(box: Box) -> Box | None:
    lo, hi = np.array(box.lo), np.array(box.hi)

    lo[0], hi[0] = max(lo[0], 0.0), min(hi[0], 2 * math.pi)
    lo[1], hi[1] = max(lo[1], q2.SPEED_BOUNDS[0]), min(hi[1], q2.SPEED_BOUNDS[1])
    if lo[0] > hi[0] or lo[1] > hi[1]:
        return None

    # Release gaps and explosion-before-impact constraints; iterate the cheap
    # interval contractor to a fixed point.
    for _ in range(3):
        lo[3] = max(lo[3], lo[2] + 1.0)
        hi[2] = min(hi[2], hi[3] - 1.0)
        lo[4] = max(lo[4], lo[3] + 1.0)
        hi[3] = min(hi[3], hi[4] - 1.0)
        for i in range(3):
            r, tau = 2 + i, 5 + i
            hi[r] = min(hi[r], T - lo[tau])
            hi[tau] = min(hi[tau], T - lo[r], TAU_MAX)
    if np.any(lo > hi + 1e-12):
        return None
    return Box(tuple(lo), tuple(hi), box.depth)


def product_interval(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    values = [x * y for x in a for y in b]
    return min(values), max(values)


def trig_interval(lo: float, hi: float, cosine: bool) -> tuple[float, float]:
    values = [math.cos(lo) if cosine else math.sin(lo), math.cos(hi) if cosine else math.sin(hi)]
    step = math.pi / 2
    first = math.ceil(lo / step)
    last = math.floor(hi / step)
    for index in range(first, last + 1):
        angle = index * step
        values.append(math.cos(angle) if cosine else math.sin(angle))
    return min(values), max(values)


def tau_height_interval(lo: float, hi: float) -> tuple[float, float]:
    """Range of 3*tau-g*tau^2/2 on [lo, hi]."""
    def value(tau: float) -> float:
        return 3.0 * tau - 0.5 * q3.G * tau * tau

    values = [value(lo), value(hi)]
    vertex = 3.0 / q3.G
    if lo <= vertex <= hi:
        values.append(value(vertex))
    return min(values), max(values)


def cloud_boxes(box: Box, times: np.ndarray, half_dt: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return center(t), axis half-widths and possible-active masks."""
    lo, hi = box.lo, box.hi
    result = []
    for i in range(3):
        r = (lo[2 + i], hi[2 + i])
        tau = (lo[5 + i], hi[5 + i])
        te = (r[0] + tau[0], r[1] + tau[1])
        travel = (lo[1] * te[0], hi[1] * te[1])
        x = product_interval(travel, trig_interval(lo[0], hi[0], True))
        y = product_interval(travel, trig_interval(lo[0], hi[0], False))
        h = tau_height_interval(*tau)
        lower = np.column_stack(
            (
                np.full(len(times), q2.UAV_0[0] + x[0]),
                np.full(len(times), q2.UAV_0[1] + y[0]),
                q2.UAV_0[2] + 3.0 * r[0] + h[0] - 3.0 * times,
            )
        )
        upper = np.column_stack(
            (
                np.full(len(times), q2.UAV_0[0] + x[1]),
                np.full(len(times), q2.UAV_0[1] + y[1]),
                q2.UAV_0[2] + 3.0 * r[1] + h[1] - 3.0 * times,
            )
        )
        center = (lower + upper) / 2
        half_width = (upper - lower) / 2
        active = (te[0] <= times + half_dt + TIME_GUARD) & (
            te[1] + geometry.SMOKE_LIFETIME >= times - half_dt - TIME_GUARD
        )
        result.append((center, half_width, active))
    return result


def witnesses(n_phi: int) -> np.ndarray:
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    return np.vstack(
        [
            np.column_stack(
                (
                    geometry.TARGET_RADIUS * np.cos(phi),
                    geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS * np.sin(phi),
                    np.full(n_phi, z),
                )
            )
            for z in (0.0, geometry.TARGET_HEIGHT)
        ]
    )


def coverage_masks(
    center: np.ndarray,
    half_width: np.ndarray,
    active: np.ndarray,
    missiles: np.ndarray,
    points: np.ndarray,
    threshold: float,
    center_cells: int,
) -> np.ndarray:
    """Possible witness masks for one ball choosing one common center subbox."""
    widths = half_width[0]
    # Allocate the power-of-two cell budget to the currently longest spatial
    # half-width.  This minimizes the subbox circumsphere greedily and is never
    # tied to one coordinate axis when two or three directions are uncertain.
    splits = np.ones(3, dtype=int)
    while int(np.prod(splits)) < center_cells and np.any(widths / splits > 1e-12):
        splits[int(np.argmax(widths / splits))] *= 2
    count = int(np.prod(splits))
    masks = np.zeros((len(missiles), count), dtype=np.uint64)
    for cell, index in enumerate(np.ndindex(tuple(int(value) for value in splits))):
        local_half = widths / splits
        local_center = center.copy()
        local_center += -widths + (2 * np.asarray(index) + 1) * local_half
        radius = float(np.linalg.norm(local_half))
        for index, point in enumerate(points):
            segment = point - missiles
            denominator = np.einsum("ti,ti->t", segment, segment)
            fraction = np.clip(
                np.einsum("ti,ti->t", local_center - missiles, segment) / denominator,
                0.0,
                1.0,
            )
            closest = missiles + fraction[:, None] * segment
            lower = np.maximum(0.0, np.linalg.norm(local_center - closest, axis=1) - radius)
            masks[active & (lower <= threshold), cell] |= np.uint64(1) << np.uint64(index)
    return masks


def upper_duration(box: Box, dt: float, points: np.ndarray, center_cells: int = 1) -> float:
    """Safe duration upper bound for all strategies in ``box``."""
    contracted = contract(box)
    if contracted is None:
        return -math.inf
    half = dt / 2
    times = np.arange(half, COVER_END, dt)
    widths = np.minimum(times + half, COVER_END) - np.maximum(times - half, 0.0)
    missiles = geometry.MISSILE_0 + geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    clouds = cloud_boxes(contracted, times, half)
    possible = np.ones(len(times), dtype=bool)
    threshold = geometry.SMOKE_RADIUS + TIME_RATE * half + DISTANCE_GUARD

    for point in points:
        segment = point - missiles
        denominator = np.einsum("ti,ti->t", segment, segment)
        any_ball = np.zeros(len(times), dtype=bool)
        segment_lo = np.minimum(missiles, point)
        segment_hi = np.maximum(missiles, point)
        for center, half_width, active in clouds:
            radius = np.linalg.norm(half_width, axis=1)
            fraction = np.clip(
                np.einsum("ti,ti->t", center - missiles, segment) / denominator,
                0.0,
                1.0,
            )
            closest = missiles + fraction[:, None] * segment
            sphere_lower = np.maximum(0.0, np.linalg.norm(center - closest, axis=1) - radius)

            # Independent AABB-to-AABB lower bound; taking the maximum of two
            # valid lower bounds is still safe and often much tighter.
            cloud_lo = center - half_width
            cloud_hi = center + half_width
            gap = np.maximum(np.maximum(segment_lo - cloud_hi, cloud_lo - segment_hi), 0.0)
            box_lower = np.linalg.norm(gap, axis=1)
            lower = np.maximum(sphere_lower, box_lower)
            any_ball |= active & (lower <= threshold)
        possible &= any_ball
        if not np.any(possible):
            return 0.0

    if center_cells > 1 and len(points) <= 63 and np.any(possible):
        masks = [
            coverage_masks(center, half_width, active, missiles, points, threshold, center_cells)
            for center, half_width, active in clouds
        ]
        full = (np.uint64(1) << np.uint64(len(points))) - np.uint64(1)
        for time_index in np.flatnonzero(possible):
            feasible = False
            for first in masks[0][time_index]:
                for second in masks[1][time_index]:
                    partial = first | second
                    if any((partial | third) == full for third in masks[2][time_index]):
                        feasible = True
                        break
                if feasible:
                    break
            possible[time_index] = feasible
    return float(np.sum(widths[possible]))


def split_index(box: Box) -> int:
    """Split the variable with the largest first-order cloud uncertainty."""
    lo, hi = np.array(box.lo), np.array(box.hi)
    widths = hi - lo
    te_hi = max(hi[2:5] + hi[5:8])
    speed_hi = hi[1]
    tau_slopes = np.maximum(
        np.abs(3.0 - q3.G * lo[5:8]),
        np.abs(3.0 - q3.G * hi[5:8]),
    )
    impact = np.r_[
        widths[0] * speed_hi * max(1.0, te_hi),
        widths[1] * max(1.0, te_hi),
        widths[2:5] * math.hypot(speed_hi, 3.0),
        widths[5:8] * np.hypot(speed_hi, tau_slopes),
    ]
    return int(np.argmax(impact))


def branch_bound(
    dt: float,
    fine_dt: float | None,
    refine_below: float,
    n_phi: int,
    target: float,
    max_nodes: int,
    center_cells: int,
    preset: str,
) -> dict[str, object]:
    points = witnesses(n_phi)
    root = contract(root_box(preset))
    assert root is not None

    def bound(box: Box) -> float:
        coarse = upper_duration(box, dt, points, center_cells)
        if fine_dt is not None and coarse <= refine_below:
            return min(coarse, upper_duration(box, fine_dt, points, center_cells))
        return coarse

    root_upper = bound(root)
    heap: list[tuple[float, int, Box]] = [(-root_upper, 0, root)]
    counter = itertools.count(1)
    pruned = infeasible = processed = 0
    deepest = 0
    while heap and processed < max_nodes:
        neg_upper, _, box = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            pruned += 1
            continue
        index = split_index(box)
        for child in box.split(index):
            child = contract(child)
            if child is None:
                infeasible += 1
                continue
            child_upper = bound(child)
            if child_upper <= target:
                pruned += 1
            else:
                heapq.heappush(heap, (-child_upper, next(counter), child))
                deepest = max(deepest, child.depth)
        processed += 1
        if processed % 100 == 0:
            print(
                f"nodes={processed},open={len(heap)},global_upper={-heap[0][0] if heap else target:.6f},"
                f"pruned={pruned},depth={deepest}",
                flush=True,
            )
    global_upper = target if not heap else max(target, -heap[0][0])
    worst = None if not heap else heap[0][2]
    return {
        "dt": dt,
        "fine_dt": fine_dt,
        "refine_below": refine_below,
        "preset": preset,
        "cover_end": COVER_END,
        "distance_guard": DISTANCE_GUARD,
        "time_guard": TIME_GUARD,
        "n_phi_per_rim": n_phi,
        "center_cells": center_cells,
        "target": target,
        "root_upper": root_upper,
        "global_upper": global_upper,
        "processed": processed,
        "open": len(heap),
        "pruned": pruned,
        "infeasible": infeasible,
        "deepest": deepest,
        "target_certified": not heap,
        "worst_box": None
        if worst is None
        else {"lo": worst.lo, "hi": worst.hi, "depth": worst.depth, "split_index": split_index(worst)},
    }


def self_test() -> None:
    points = witnesses(16)
    incumbent = point_box()
    upper = upper_duration(incumbent, 0.01, points)
    assert upper >= 7.650405706
    parent = contract(root_box())
    assert parent is not None
    parent_upper = upper_duration(parent, 0.05, points)
    assert 0.0 <= parent_upper <= COVER_END + 1e-9
    for child in parent.split(0):
        value = upper_duration(child, 0.05, points)
        assert -math.inf <= value <= COVER_END + 1e-9
    assert contract(Box((0.0,) * 8, (1.0,) * 8)) is None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fine-dt", type=float)
    parser.add_argument("--refine-below", type=float, default=11.0)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--target", type=float, default=8.0)
    parser.add_argument("--max-nodes", type=int, default=2_000)
    parser.add_argument("--center-cells", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--preset", choices=["full", "immediate"], default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q3_upper_branch_bound.json"),
    )
    args = parser.parse_args()
    self_test()
    result = branch_bound(
        args.dt,
        args.fine_dt,
        args.refine_below,
        args.n_phi,
        args.target,
        args.max_nodes,
        args.center_cells,
        args.preset,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
