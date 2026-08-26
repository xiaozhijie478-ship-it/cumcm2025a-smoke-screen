"""Safe outer branch bound for the complete Q4 control domain.

The relaxation is deliberately optimistic.  Inside one parameter box, every
time bin and every target witness may choose whichever smoke ball can possibly
intersect its finite missile-to-witness segment.  Consequently, a duration
discarded by this program is impossible for the physical Q4 problem as well.
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
import q3_upper_branch_bound as q3ub
import q4_optimize as q4


T = geometry.MISSILE_HIT_TIME
TIME_RATE = geometry.MISSILE_SPEED + geometry.SMOKE_SINK_SPEED
DISTANCE_GUARD = 1e-8
TIME_GUARD = 1e-12
NAMES = tuple(q4.UAVS)
SINGLE_DURATION_UPPERS = {"FY1": 4.589, "FY2": 4.000, "FY3": 3.250}
TARGET_LOWER = np.array(
    [
        -geometry.TARGET_RADIUS,
        geometry.TARGET_CENTER_XY[1] - geometry.TARGET_RADIUS,
        0.0,
    ]
)
TARGET_UPPER = np.array(
    [
        geometry.TARGET_RADIUS,
        geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS,
        geometry.TARGET_HEIGHT,
    ]
)


@dataclass(frozen=True)
class Box:
    """Independent (theta, speed, explosion time, delay) boxes for FY1--FY3."""

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


def root_box(
    names: tuple[str, ...] = NAMES,
    active_time_range: tuple[float, float] | None = None,
) -> Box:
    lo: list[float] = []
    hi: list[float] = []
    for name in names:
        free_fall = math.sqrt(2 * q4.UAVS[name][2] / q4.G)
        explosion_lo, explosion_hi = 0.0, T
        if active_time_range is not None:
            explosion_lo = max(
                0.0, active_time_range[0] - geometry.SMOKE_LIFETIME
            )
            explosion_hi = min(T, active_time_range[1])
        lo.extend((0.0, q2.SPEED_BOUNDS[0], explosion_lo, 0.0))
        hi.extend((2 * math.pi, q2.SPEED_BOUNDS[1], explosion_hi, free_fall))
    return Box(tuple(lo), tuple(hi))


def point_box(names: tuple[str, ...] = NAMES) -> Box:
    values: list[float] = []
    by_name = {item.name: item for item in q4.final_total_candidate()}
    for name in names:
        strategy = by_name[name]
        values.extend(
            (strategy.theta, strategy.speed, strategy.explosion_time, strategy.delay)
        )
    return Box(tuple(values), tuple(values))


def contract(box: Box, names: tuple[str, ...] = NAMES) -> Box | None:
    lo, hi = np.array(box.lo), np.array(box.hi)
    for index, name in enumerate(names):
        base = 4 * index
        theta, speed, explosion, delay = base, base + 1, base + 2, base + 3
        lo[theta], hi[theta] = max(lo[theta], 0.0), min(hi[theta], 2 * math.pi)
        lo[speed], hi[speed] = (
            max(lo[speed], q2.SPEED_BOUNDS[0]),
            min(hi[speed], q2.SPEED_BOUNDS[1]),
        )
        free_fall = math.sqrt(2 * q4.UAVS[name][2] / q4.G)
        lo[explosion], lo[delay] = max(lo[explosion], 0.0), max(lo[delay], 0.0)
        hi[explosion] = min(hi[explosion], T)
        hi[delay] = min(hi[delay], free_fall, hi[explosion])
        lo[explosion] = max(lo[explosion], lo[delay])
    if np.any(lo > hi + 1e-12):
        return None
    return Box(tuple(lo), tuple(hi), box.depth)


def cloud_boxes(
    box: Box,
    times: np.ndarray,
    half_dt: float,
    names: tuple[str, ...] = NAMES,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return center boxes and possible-active masks for all three clouds."""
    lo, hi = box.lo, box.hi
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, name in enumerate(names):
        base = 4 * index
        theta = (lo[base], hi[base])
        speed = (lo[base + 1], hi[base + 1])
        explosion = (lo[base + 2], hi[base + 2])
        delay = (lo[base + 3], hi[base + 3])
        travel = q3ub.product_interval(speed, explosion)
        x = q3ub.product_interval(travel, q3ub.trig_interval(*theta, cosine=True))
        y = q3ub.product_interval(travel, q3ub.trig_interval(*theta, cosine=False))
        uav = q4.UAVS[name]
        lower = np.column_stack(
            (
                np.full(len(times), uav[0] + x[0]),
                np.full(len(times), uav[1] + y[0]),
                uav[2]
                + 3.0 * explosion[0]
                - 0.5 * q4.G * delay[1] ** 2
                - 3.0 * times,
            )
        )
        upper = np.column_stack(
            (
                np.full(len(times), uav[0] + x[1]),
                np.full(len(times), uav[1] + y[1]),
                uav[2]
                + 3.0 * explosion[1]
                - 0.5 * q4.G * delay[0] ** 2
                - 3.0 * times,
            )
        )
        center = (lower + upper) / 2
        half_width = (upper - lower) / 2
        active = (explosion[0] <= times + half_dt + TIME_GUARD) & (
            explosion[1] + geometry.SMOKE_LIFETIME
            >= times - half_dt - TIME_GUARD
        )
        result.append((center, half_width, active))
    return result


def upper_duration(
    box: Box,
    dt: float,
    points: np.ndarray,
    center_cells: int = 1,
    names: tuple[str, ...] = NAMES,
    time_range: tuple[float, float] = (0.0, T),
    connected_single_cap: bool = False,
) -> float:
    """Safe cumulative-duration upper bound for every strategy in ``box``."""
    contracted = contract(box, names)
    if contracted is None:
        return -math.inf
    half = dt / 2
    time_start, time_end = time_range
    times = np.arange(time_start + half, time_end, dt)
    widths = np.minimum(times + half, time_end) - np.maximum(
        times - half, time_start
    )
    missiles = (
        geometry.MISSILE_0
        + geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    )
    clouds = cloud_boxes(contracted, times, half, names)
    possible = np.ones(len(times), dtype=bool)
    threshold = geometry.SMOKE_RADIUS + TIME_RATE * half + DISTANCE_GUARD

    # A ball that is essential to joint-only coverage must touch at least one
    # finite missile-to-target segment.  The union of all such segments lies
    # inside this time-dependent AABB, so AABB separation gives a safe (and
    # deliberately optimistic) possible-contribution mask.
    segment_union_lo = np.minimum(missiles, TARGET_LOWER)
    segment_union_hi = np.maximum(missiles, TARGET_UPPER)
    contribution_possible: list[np.ndarray] = []
    for center, half_width, active in clouds:
        cloud_lo, cloud_hi = center - half_width, center + half_width
        gap = np.maximum(
            np.maximum(segment_union_lo - cloud_hi, cloud_lo - segment_union_hi),
            0.0,
        )
        contribution_possible.append(
            active & (np.linalg.norm(gap, axis=1) <= threshold)
        )

    for point in points:
        segment = point - missiles
        denominator = np.einsum("ti,ti->t", segment, segment)
        segment_lo = np.minimum(missiles, point)
        segment_hi = np.maximum(missiles, point)
        any_ball = np.zeros(len(times), dtype=bool)
        for center, half_width, active in clouds:
            radius = np.linalg.norm(half_width, axis=1)
            fraction = np.clip(
                np.einsum("ti,ti->t", center - missiles, segment) / denominator,
                0.0,
                1.0,
            )
            closest = missiles + fraction[:, None] * segment
            sphere_lower = np.maximum(
                0.0, np.linalg.norm(center - closest, axis=1) - radius
            )
            cloud_lo, cloud_hi = center - half_width, center + half_width
            gap = np.maximum(
                np.maximum(segment_lo - cloud_hi, cloud_lo - segment_hi), 0.0
            )
            lower = np.maximum(sphere_lower, np.linalg.norm(gap, axis=1))
            any_ball |= active & (lower <= threshold)
        possible &= any_ball
        if not np.any(possible):
            return 0.0

    if center_cells > 1 and len(points) <= 63 and np.any(possible):
        masks = [
            q3ub.coverage_masks(
                center, half_width, active, missiles, points, threshold, center_cells
            )
            for center, half_width, active in clouds
        ]
        full = (np.uint64(1) << np.uint64(len(points))) - np.uint64(1)
        for time_index in np.flatnonzero(possible):
            possible[time_index] = any(
                np.bitwise_or.reduce(np.asarray(choice, dtype=np.uint64)) == full
                for choice in itertools.product(
                    *(item[time_index] for item in masks)
                )
            )
    # A physical strategy has only three radius-10 balls, each active for at
    # most 20 s.  This cap remains valid even though the possible-active masks
    # above independently relax the explosion time in every bin.
    upper = min(
        float(np.sum(widths[possible])),
        len(names) * geometry.SMOKE_LIFETIME,
        time_end - time_start,
    )
    if connected_single_cap and len(names) <= 2:
        if len(names) == 1:
            upper = min(upper, SINGLE_DURATION_UPPERS[names[0]])
        else:
            joint_possible = contribution_possible[0] & contribution_possible[1]
            joint_only_upper = float(np.sum(widths[joint_possible]))
            upper = min(
                upper,
                sum(SINGLE_DURATION_UPPERS[name] for name in names)
                + joint_only_upper,
            )
    return upper


def split_index(box: Box) -> int:
    lo, hi = np.array(box.lo), np.array(box.hi)
    explosion_indices = np.arange(2, len(lo), 4)
    explosion_widths = hi[explosion_indices] - lo[explosion_indices]
    if np.max(explosion_widths) > 2.5:
        return int(explosion_indices[int(np.argmax(explosion_widths))])
    impacts = np.empty(len(lo))
    for index in range(len(lo) // 4):
        base = 4 * index
        widths = hi[base : base + 4] - lo[base : base + 4]
        explosion_hi = hi[base + 2]
        speed_hi = hi[base + 1]
        tau_slope = max(
            abs(3.0 - q4.G * lo[base + 3]),
            abs(3.0 - q4.G * hi[base + 3]),
        )
        impacts[base : base + 4] = (
            widths[0] * speed_hi * max(1.0, explosion_hi),
            widths[1] * max(1.0, explosion_hi),
            widths[2] * math.hypot(speed_hi, 3.0),
            widths[3] * math.hypot(speed_hi, tau_slope),
        )
    return int(np.argmax(impacts))


def branch_bound(
    dt: float,
    n_phi: int,
    target: float,
    max_nodes: int,
    center_cells: int,
    names: tuple[str, ...] = NAMES,
    time_range: tuple[float, float] = (0.0, T),
    connected_single_cap: bool = False,
) -> dict[str, object]:
    points = q3ub.witnesses(n_phi)
    active_time_range = None
    if connected_single_cap and len(names) <= 2:
        if target < max(SINGLE_DURATION_UPPERS[name] for name in names):
            raise ValueError(
                "target must dominate every excluded inactive-ball single upper"
            )
        active_time_range = time_range
    root = contract(root_box(names, active_time_range), names)
    assert root is not None
    root_upper = upper_duration(
        root,
        dt,
        points,
        center_cells,
        names,
        time_range,
        connected_single_cap,
    )
    heap: list[tuple[float, int, Box]] = [(-root_upper, 0, root)]
    counter = itertools.count(1)
    processed = pruned = infeasible = deepest = 0
    while heap and processed < max_nodes:
        neg_upper, _, box = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            pruned += 1
            continue
        for child in box.split(split_index(box)):
            child = contract(child, names)
            if child is None:
                infeasible += 1
                continue
            child_upper = upper_duration(
                child,
                dt,
                points,
                center_cells,
                names,
                time_range,
                connected_single_cap,
            )
            if child_upper <= target:
                pruned += 1
            else:
                heapq.heappush(heap, (-child_upper, next(counter), child))
                deepest = max(deepest, child.depth)
        processed += 1
        if processed % 100 == 0:
            print(
                f"nodes={processed},open={len(heap)},"
                f"global_upper={-heap[0][0] if heap else target:.6f},"
                f"pruned={pruned},depth={deepest}",
                flush=True,
            )
    global_upper = target if not heap else max(target, -heap[0][0])
    worst = None if not heap else heap[0][2]
    return {
        "dt": dt,
        "n_phi_per_rim": n_phi,
        "center_cells": center_cells,
        "names": names,
        "time_range": time_range,
        "connected_single_cap": connected_single_cap,
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
        else {"lo": worst.lo, "hi": worst.hi, "depth": worst.depth},
    }


def self_test() -> None:
    points = q3ub.witnesses(8)
    incumbent_upper = upper_duration(point_box(), 0.02, points, 4)
    assert incumbent_upper + 1e-10 >= 11.735130825
    root = contract(root_box())
    assert root is not None
    root_upper = upper_duration(root, 0.10, points, 1)
    assert 0.0 <= root_upper <= T + 1e-9
    pair = ("FY1", "FY2")
    pair_root = contract(root_box(pair), pair)
    assert pair_root is not None
    pair_upper = upper_duration(
        pair_root, 0.10, points, 2, pair, (0.0, 13.942236250)
    )
    assert 0.0 <= pair_upper <= 13.942236250 + 1e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--n-phi", type=int, default=8)
    parser.add_argument("--target", type=float, default=13.0)
    parser.add_argument("--max-nodes", type=int, default=2_000)
    parser.add_argument("--center-cells", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--names", default=",".join(NAMES))
    parser.add_argument("--time-range", default=f"0,{T}")
    parser.add_argument("--connected-single-cap", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q4_upper_branch_bound.json"),
    )
    args = parser.parse_args()
    self_test()
    names = tuple(args.names.split(","))
    if not names or len(set(names)) != len(names) or any(name not in q4.UAVS for name in names):
        raise ValueError("--names must be a comma-separated subset of FY1,FY2,FY3")
    time_range = tuple(float(item) for item in args.time_range.split(","))
    if len(time_range) != 2 or not 0.0 <= time_range[0] < time_range[1] <= T:
        raise ValueError("--time-range must be START,END inside the missile flight")
    result = branch_bound(
        args.dt,
        args.n_phi,
        args.target,
        args.max_nodes,
        args.center_cells,
        names,
        time_range,
        args.connected_single_cap,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
