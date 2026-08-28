"""Safe Q3 outer bound in shared velocity/explosion-time coordinates.

The parameter box is ``(ux, uy, e1, e2, e3, tau1, tau2, tau3)``.  Here
``(ux, uy)`` is the UAV's common horizontal velocity, ``ei`` is bomb ``i``'s
explosion time, and ``taui`` is its fuse delay.  Release times are recovered
as ``ri = ei - taui``.  This removes the avoidable speed/heading correlation
loss in the original ``(theta, speed, release, delay)`` interval model.
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
import q3_upper_branch_bound as legacy


T = legacy.T
COVER_END = legacy.COVER_END
TAU_MAX = legacy.TAU_MAX
DISTANCE_GUARD = legacy.DISTANCE_GUARD
TIME_GUARD = legacy.TIME_GUARD
TIME_RATE = legacy.TIME_RATE
SPEED_MIN, SPEED_MAX = q2.SPEED_BOUNDS
CONTRACT_GUARD = 1e-10
CELL_GUARD = 1e-4
EVENT_TIME_EPS = 1e-3


@dataclass(frozen=True)
class Box:
    """Interval box for ``(ux, uy, e1, e2, e3, tau1, tau2, tau3)``."""

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


def velocity_component_bounds(
    theta_range: tuple[float, float] | None,
    speed_range: tuple[float, float] = q2.SPEED_BOUNDS,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned enclosure of an annular heading sector."""
    theta_lo, theta_hi = theta_range or (0.0, 2 * math.pi)
    return (
        legacy.product_interval(
            speed_range, legacy.trig_interval(theta_lo, theta_hi, True)
        ),
        legacy.product_interval(
            speed_range, legacy.trig_interval(theta_lo, theta_hi, False)
        ),
    )


def root_box(
    preset: str = "full",
    theta_range: tuple[float, float] | None = None,
) -> Box:
    (ux_lo, ux_hi), (uy_lo, uy_hi) = velocity_component_bounds(theta_range)
    if preset == "immediate":
        return Box(
            (ux_lo, uy_lo, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
            (ux_hi, uy_hi, 0.0, T, T, 0.0, TAU_MAX, TAU_MAX),
        )
    if preset == "two":
        # Any physical two-bomb strategy embeds into this domain by releasing
        # an inert third bomb at impact: e3=T and tau3=0.
        return Box(
            (ux_lo, uy_lo, 0.0, 1.0, T, 0.0, 0.0, 0.0),
            (ux_hi, uy_hi, T, T, T, TAU_MAX, TAU_MAX, 0.0),
        )
    return Box(
        (ux_lo, uy_lo, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
        (ux_hi, uy_hi, T, T, T, TAU_MAX, TAU_MAX, TAU_MAX),
    )


def strategy_box(strategies: list[q2.Strategy]) -> Box:
    first = strategies[0]
    ux = first.speed * math.cos(first.theta)
    uy = first.speed * math.sin(first.theta)
    values = (
        ux,
        uy,
        *(item.explosion_time for item in strategies),
        *(item.delay for item in strategies),
    )
    return Box(tuple(values), tuple(values))


def _min_abs(lo: float, hi: float) -> float:
    return 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))


def _max_abs(lo: float, hi: float) -> float:
    return max(abs(lo), abs(hi))


def _contract_linear_ge(
    lo: np.ndarray,
    hi: np.ndarray,
    ax: float,
    ay: float,
) -> None:
    """Hull contraction for ``ax*ux + ay*uy >= 0``."""
    if abs(ax) > 1e-15:
        bounds = [(-ay * value) / ax for value in (lo[1], hi[1])]
        if ax > 0:
            lo[0] = max(lo[0], min(bounds))
        else:
            hi[0] = min(hi[0], max(bounds))
    if abs(ay) > 1e-15:
        bounds = [(-ax * value) / ay for value in (lo[0], hi[0])]
        if ay > 0:
            lo[1] = max(lo[1], min(bounds))
        else:
            hi[1] = min(hi[1], max(bounds))


def _contract_velocity(
    lo: np.ndarray,
    hi: np.ndarray,
    theta_range: tuple[float, float] | None,
) -> bool:
    """Contract the velocity rectangle against speed and heading constraints."""
    for _ in range(4):
        # Upper speed disk.
        for index, other in ((0, 1), (1, 0)):
            other_min = _min_abs(lo[other], hi[other])
            limit = math.sqrt(max(0.0, SPEED_MAX**2 - other_min**2))
            lo[index], hi[index] = max(lo[index], -limit), min(hi[index], limit)

        # Lower speed radius.  A sign-definite interval can be contracted
        # without branching; a zero-crossing interval keeps both possibilities.
        for index, other in ((0, 1), (1, 0)):
            other_max = _max_abs(lo[other], hi[other])
            required = math.sqrt(max(0.0, SPEED_MIN**2 - other_max**2))
            if hi[index] <= 0.0:
                hi[index] = min(hi[index], -required)
            elif lo[index] >= 0.0:
                lo[index] = max(lo[index], required)

        if theta_range is not None and theta_range[1] - theta_range[0] <= math.pi:
            theta_lo, theta_hi = theta_range
            # cross(d_lo, u) >= 0 and cross(u, d_hi) >= 0.
            _contract_linear_ge(lo, hi, -math.sin(theta_lo), math.cos(theta_lo))
            _contract_linear_ge(lo, hi, math.sin(theta_hi), -math.cos(theta_hi))

        if np.any(lo[:2] > hi[:2] + CONTRACT_GUARD):
            return False
        min_norm2 = _min_abs(lo[0], hi[0]) ** 2 + _min_abs(lo[1], hi[1]) ** 2
        max_norm2 = _max_abs(lo[0], hi[0]) ** 2 + _max_abs(lo[1], hi[1]) ** 2
        if min_norm2 > SPEED_MAX**2 + CONTRACT_GUARD:
            return False
        if max_norm2 < SPEED_MIN**2 - CONTRACT_GUARD:
            return False
    return True


def contract(
    box: Box,
    required_active: tuple[int, ...] = (),
    theta_range: tuple[float, float] | None = None,
    explosion_order: tuple[int, ...] = (),
) -> Box | None:
    lo, hi = np.array(box.lo, dtype=float), np.array(box.hi, dtype=float)
    if not _contract_velocity(lo, hi, theta_range):
        return None

    lo[2:5] = np.maximum(lo[2:5], 0.0)
    hi[2:5] = np.minimum(hi[2:5], T)
    lo[5:8] = np.maximum(lo[5:8], 0.0)
    hi[5:8] = np.minimum(hi[5:8], TAU_MAX)

    # Fixed-point hull contraction for ri=ei-taui >= 0 and consecutive
    # release gaps r_(i+1)-r_i >= 1.
    for _ in range(8):
        for i in range(3):
            e, tau = 2 + i, 5 + i
            lo[e] = max(lo[e], lo[tau])
            hi[tau] = min(hi[tau], hi[e])
            if i in required_active:
                hi[e] = min(hi[e], COVER_END)
        for left, right, gap in ((0, 1, 1.0), (1, 2, 1.0), (0, 2, 2.0)):
            e_l, e_r = 2 + left, 2 + right
            t_l, t_r = 5 + left, 5 + right
            lo[e_r] = max(lo[e_r], gap - hi[t_l] + lo[t_r] + lo[e_l])
            lo[t_l] = max(lo[t_l], gap - hi[e_r] + lo[t_r] + lo[e_l])
            hi[t_r] = min(hi[t_r], hi[e_r] + hi[t_l] - lo[e_l] - gap)
            hi[e_l] = min(hi[e_l], hi[e_r] + hi[t_l] - lo[t_r] - gap)
        for left, right in zip(explosion_order, explosion_order[1:]):
            hi[2 + left] = min(hi[2 + left], hi[2 + right])
            lo[2 + right] = max(lo[2 + right], lo[2 + left])
        if explosion_order:
            rank = {bomb: place for place, bomb in enumerate(explosion_order)}
            for earlier in range(3):
                for later in range(earlier + 1, 3):
                    if (
                        earlier in rank
                        and later in rank
                        and rank[later] < rank[earlier]
                    ):
                        gap = float(later - earlier)
                        lo[5 + earlier] = max(lo[5 + earlier], lo[5 + later] + gap)
                        hi[5 + later] = min(hi[5 + later], hi[5 + earlier] - gap)
        if np.any(lo > hi + CONTRACT_GUARD):
            return None
    return Box(tuple(lo), tuple(hi), box.depth)


def cloud_boxes(
    box: Box,
    times: np.ndarray,
    half_dt: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return interval cloud centers and possible-active masks."""
    result = []
    for i in range(3):
        e = (box.lo[2 + i], box.hi[2 + i])
        tau = (box.lo[5 + i], box.hi[5 + i])
        x = legacy.product_interval((box.lo[0], box.hi[0]), e)
        y = legacy.product_interval((box.lo[1], box.hi[1]), e)
        z_lo = q2.UAV_0[2] + 3.0 * e[0] - 0.5 * q3.G * tau[1] ** 2
        z_hi = q2.UAV_0[2] + 3.0 * e[1] - 0.5 * q3.G * tau[0] ** 2
        lower = np.column_stack(
            (
                np.full(len(times), q2.UAV_0[0] + x[0]),
                np.full(len(times), q2.UAV_0[1] + y[0]),
                z_lo - geometry.SMOKE_SINK_SPEED * times,
            )
        )
        upper = np.column_stack(
            (
                np.full(len(times), q2.UAV_0[0] + x[1]),
                np.full(len(times), q2.UAV_0[1] + y[1]),
                z_hi - geometry.SMOKE_SINK_SPEED * times,
            )
        )
        active = (e[0] <= times + half_dt + TIME_GUARD) & (
            e[1] + geometry.SMOKE_LIFETIME >= times - half_dt - TIME_GUARD
        )
        result.append(((lower + upper) / 2, (upper - lower) / 2, active))
    return result


def parameter_cells_compatible(
    box: Box,
    event_boxes: list[list[Box | None]],
    event_clouds: list[list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]],
    event_partitions: list[list[list[tuple[np.ndarray, np.ndarray]] | None]],
    choices: tuple[tuple[int, int], ...],
    relevant: list[int],
    time0: float,
) -> bool:
    """Necessary common-velocity and release-gap test for spatial cells."""
    ux_lo, uy_lo = box.lo[:2]
    ux_hi, uy_hi = box.hi[:2]
    releases: dict[int, list[float]] = {}
    for i, (event_cell, center_cell) in zip(relevant, choices):
        subbox = event_boxes[i][event_cell]
        cloud = event_clouds[i][event_cell]
        partitions = event_partitions[i][event_cell]
        assert subbox is not None and cloud is not None and partitions is not None
        center, _, _ = cloud
        offset, local_half = partitions[center_cell]
        invariant = center[0] + offset + np.array(
            [0.0, 0.0, geometry.SMOKE_SINK_SPEED * time0]
        )
        rect_lo = invariant - local_half
        rect_hi = invariant + local_half
        e_lo, e_hi = subbox.lo[2 + i], subbox.hi[2 + i]

        # C_xy-U0_xy = u_xy*e.  Near e=0 the quotient is ill-conditioned;
        # skipping this optional filter is safe and mirrors the legacy guard.
        if e_lo > EVENT_TIME_EPS:
            for axis in range(2):
                displacement = (
                    rect_lo[axis] - q2.UAV_0[axis],
                    rect_hi[axis] - q2.UAV_0[axis],
                )
                values = [value / event for value in displacement for event in (e_lo, e_hi)]
                derived_lo, derived_hi = min(values), max(values)
                if axis == 0:
                    ux_lo = max(ux_lo, derived_lo - CELL_GUARD)
                    ux_hi = min(ux_hi, derived_hi + CELL_GUARD)
                else:
                    uy_lo = max(uy_lo, derived_lo - CELL_GUARD)
                    uy_hi = min(uy_hi, derived_hi + CELL_GUARD)
        if ux_lo > ux_hi + CELL_GUARD or uy_lo > uy_hi + CELL_GUARD:
            return False

        # C_z+3t = U0_z+3e-g*tau^2/2.  Recover an outer delay interval from
        # the chosen vertical cell, then an outer release-time interval.
        energy_lo = q2.UAV_0[2] + 3.0 * e_lo - rect_hi[2] - CELL_GUARD
        energy_hi = q2.UAV_0[2] + 3.0 * e_hi - rect_lo[2] + CELL_GUARD
        if energy_hi < 0.0:
            return False
        tau_lo = math.sqrt(max(0.0, 2.0 * energy_lo / q3.G))
        tau_hi = math.sqrt(max(0.0, 2.0 * energy_hi / q3.G))
        tau_lo = max(tau_lo, subbox.lo[5 + i])
        tau_hi = min(tau_hi, subbox.hi[5 + i])
        if tau_lo > tau_hi + CELL_GUARD:
            return False
        releases[i] = [e_lo - tau_hi - CELL_GUARD, e_hi - tau_lo + CELL_GUARD]

    ordered = sorted(releases)
    for _ in range(len(ordered)):
        for left, right in zip(ordered, ordered[1:]):
            gap = float(right - left)
            releases[right][0] = max(releases[right][0], releases[left][0] + gap)
            releases[left][1] = min(releases[left][1], releases[right][1] - gap)
    return all(lo <= hi + CELL_GUARD for lo, hi in releases.values())


def upper_duration(
    box: Box,
    dt: float,
    points: np.ndarray,
    center_cells: int = 1,
    single_cumulative_cap: float | None = None,
    consistent_center_cells: bool = False,
    required_active: tuple[int, ...] = (),
    event_cells: int = 1,
    joint_cell_consistency: bool = False,
    theta_range: tuple[float, float] | None = None,
    explosion_order: tuple[int, ...] = (),
) -> float:
    """Safe duration upper bound for every physical strategy in ``box``."""
    box = contract(box, required_active, theta_range, explosion_order)
    if box is None:
        return -math.inf
    half = dt / 2
    times = np.arange(half, COVER_END, dt)
    widths = np.minimum(times + half, COVER_END) - np.maximum(times - half, 0.0)
    missiles = (
        geometry.MISSILE_0
        + geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    )
    clouds = cloud_boxes(box, times, half)
    possible = np.ones(len(times), dtype=bool)
    threshold = geometry.SMOKE_RADIUS + TIME_RATE * half + DISTANCE_GUARD

    for point in points:
        segment = point - missiles
        denominator = np.einsum("ti,ti->t", segment, segment)
        segment_lo = np.minimum(missiles, point)
        segment_hi = np.maximum(missiles, point)
        any_ball = np.zeros(len(times), dtype=bool)
        for center, half_width, active in clouds:
            fraction = np.clip(
                np.einsum("ti,ti->t", center - missiles, segment) / denominator,
                0.0,
                1.0,
            )
            closest = missiles + fraction[:, None] * segment
            sphere_lower = np.maximum(
                0.0,
                np.linalg.norm(center - closest, axis=1)
                - np.linalg.norm(half_width, axis=1),
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

    masks: list[np.ndarray] | None = None
    consistent_upper: float | None = None
    if (center_cells > 1 or event_cells > 1) and len(points) <= 63:
        masks = [
            legacy.coverage_masks(
                center, half_width, active, missiles, points, threshold, center_cells
            )
            for center, half_width, active in clouds
        ]
        full = (np.uint64(1) << np.uint64(len(points))) - np.uint64(1)
        if consistent_center_cells:
            consistent_upper = 0.0
            for cells in itertools.product(*(range(mask.shape[1]) for mask in masks)):
                combined = masks[0][:, cells[0]].copy()
                for i in range(1, len(masks)):
                    combined |= masks[i][:, cells[i]]
                feasible = possible & (combined == full)
                consistent_upper = max(
                    consistent_upper, float(np.sum(widths[feasible]))
                )

    upper = (
        consistent_upper
        if consistent_upper is not None
        else float(np.sum(widths[possible]))
    )
    if event_cells > 1 and masks is not None:
        full = (np.uint64(1) << np.uint64(len(points))) - np.uint64(1)
        event_masks: list[list[np.ndarray]] = []
        event_boxes: list[list[Box | None]] = []
        event_clouds: list[
            list[tuple[np.ndarray, np.ndarray, np.ndarray] | None]
        ] = []
        event_partitions: list[
            list[list[tuple[np.ndarray, np.ndarray]] | None]
        ] = []
        for i in range(3):
            edges = np.linspace(box.lo[2 + i], box.hi[2 + i], event_cells + 1)
            mask_choices = []
            box_choices: list[Box | None] = []
            cloud_choices: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
            partition_choices: list[list[tuple[np.ndarray, np.ndarray]] | None] = []
            for cell in range(event_cells):
                sub_lo, sub_hi = list(box.lo), list(box.hi)
                sub_lo[2 + i], sub_hi[2 + i] = edges[cell], edges[cell + 1]
                subbox = contract(
                    Box(tuple(sub_lo), tuple(sub_hi), box.depth),
                    required_active,
                    theta_range,
                    explosion_order,
                )
                if subbox is None:
                    mask_choices.append(np.zeros((len(times), 1), dtype=np.uint64))
                    box_choices.append(None)
                    cloud_choices.append(None)
                    partition_choices.append(None)
                    continue
                center, half_width, active = cloud_boxes(subbox, times, half)[i]
                cloud = (center, half_width, active)
                mask_choices.append(
                    legacy.coverage_masks(
                        center,
                        half_width,
                        active,
                        missiles,
                        points,
                        threshold,
                        center_cells,
                    )
                )
                box_choices.append(subbox)
                cloud_choices.append(cloud)
                partition_choices.append(
                    legacy.center_partition(half_width[0], center_cells)
                )
            event_masks.append(mask_choices)
            event_boxes.append(box_choices)
            event_clouds.append(cloud_choices)
            event_partitions.append(partition_choices)
        event_upper = 0.0
        for cells in itertools.product(range(event_cells), repeat=3):
            combined = np.bitwise_or.reduce(event_masks[0][cells[0]], axis=1)
            combined |= np.bitwise_or.reduce(event_masks[1][cells[1]], axis=1)
            combined |= np.bitwise_or.reduce(event_masks[2][cells[2]], axis=1)
            event_upper = max(
                event_upper, float(np.sum(widths[combined == full]))
            )
        upper = min(upper, event_upper)

        if joint_cell_consistency:
            relevant = [
                i
                for i, choices in enumerate(event_masks)
                if any(np.any(mask) for mask in choices)
            ]
            if not relevant:
                return 0.0
            paired_upper = 0.0
            choices_per_ball = [
                [
                    (event_cell, center_cell)
                    for event_cell, event_choice in enumerate(event_masks[i])
                    for center_cell in range(event_choice.shape[1])
                    if np.any(event_choice[:, center_cell])
                ]
                for i in relevant
            ]
            for choices in itertools.product(*choices_per_ball):
                if not parameter_cells_compatible(
                    box,
                    event_boxes,
                    event_clouds,
                    event_partitions,
                    choices,
                    relevant,
                    float(times[0]),
                ):
                    continue
                event_cell, center_cell = choices[0]
                first = relevant[0]
                combined = event_masks[first][event_cell][:, center_cell].copy()
                for position in range(1, len(relevant)):
                    i = relevant[position]
                    event_cell, center_cell = choices[position]
                    combined |= event_masks[i][event_cell][:, center_cell]
                paired_upper = max(
                    paired_upper, float(np.sum(widths[combined == full]))
                )
            upper = min(upper, paired_upper)

    if single_cumulative_cap is not None:
        possible_balls = sum(bool(np.any(active)) for _, _, active in clouds)
        upper = min(upper, possible_balls * single_cumulative_cap)
    return upper


def split_index(box: Box) -> int:
    """Split the variable with the largest first-order cloud displacement."""
    lo, hi = np.array(box.lo), np.array(box.hi)
    widths = hi - lo
    e_hi = max(1.0, float(np.max(hi[2:5])))
    velocity_hi = math.hypot(
        _max_abs(lo[0], hi[0]), _max_abs(lo[1], hi[1])
    )
    impact = np.r_[
        widths[:2] * e_hi,
        # Explosion-time splits tighten horizontal position, vertical position,
        # activity, and the conditioned event masks at once.  Delay splits only
        # tighten height/release feasibility, whose spatial effect is already
        # partly captured by center cells.
        2.0 * widths[2:5] * math.hypot(velocity_hi, geometry.SMOKE_SINK_SPEED),
        0.5 * widths[5:8] * q3.G * np.maximum(1.0, hi[5:8]),
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
    single_cumulative_cap: float | None = None,
    consistent_center_cells: bool = False,
    required_active: tuple[int, ...] = (),
    event_cells: int = 1,
    theta_range: tuple[float, float] | None = None,
    joint_cell_consistency: bool = False,
    strong_branch_below: float | None = None,
    explosion_order: tuple[int, ...] = (),
) -> dict[str, object]:
    points = legacy.witnesses(n_phi)
    root = contract(
        root_box(preset, theta_range), required_active, theta_range, explosion_order
    )
    assert root is not None

    def bound(box: Box) -> float:
        coarse = upper_duration(
            box,
            dt,
            points,
            center_cells,
            single_cumulative_cap,
            consistent_center_cells,
            required_active,
            event_cells,
            joint_cell_consistency,
            theta_range,
            explosion_order,
        )
        if fine_dt is not None and coarse <= refine_below:
            return min(
                coarse,
                upper_duration(
                    box,
                    fine_dt,
                    points,
                    center_cells,
                    single_cumulative_cap,
                    consistent_center_cells,
                    required_active,
                    event_cells,
                    joint_cell_consistency,
                    theta_range,
                    explosion_order,
                ),
            )
        return coarse

    root_upper = bound(root)
    root_pruned = root_upper <= target
    heap: list[tuple[float, int, Box]] = (
        [] if root_pruned else [(-root_upper, 0, root)]
    )
    counter = itertools.count(1)
    pruned, infeasible, processed, deepest = int(root_pruned), 0, 0, 0
    while heap and processed < max_nodes:
        neg_upper, _, box = heapq.heappop(heap)
        upper = -neg_upper
        if upper <= target:
            pruned += 1
            continue

        def children_for(index: int) -> tuple[list[tuple[Box, float]], int]:
            children = []
            rejected = 0
            for child in box.split(index):
                child = contract(
                    child, required_active, theta_range, explosion_order
                )
                if child is None:
                    rejected += 1
                else:
                    children.append((child, min(upper, bound(child))))
            return children, rejected

        index = split_index(box)
        children, rejected = children_for(index)
        if strong_branch_below is not None and upper <= strong_branch_below:
            best_key = (
                max((value for _, value in children), default=-math.inf),
                sum(value for _, value in children),
                index,
            )
            for candidate in range(8):
                if candidate == index or box.hi[candidate] - box.lo[candidate] <= 1e-12:
                    continue
                candidate_children, candidate_rejected = children_for(candidate)
                key = (
                    max((value for _, value in candidate_children), default=-math.inf),
                    sum(value for _, value in candidate_children),
                    candidate,
                )
                if key < best_key:
                    best_key = key
                    index = candidate
                    children = candidate_children
                    rejected = candidate_rejected
                    if best_key[0] <= target:
                        break
        infeasible += rejected
        for child, child_upper in children:
            if child_upper <= target:
                pruned += 1
            else:
                heapq.heappush(heap, (-child_upper, next(counter), child))
                deepest = max(deepest, child.depth)
        processed += 1
        if processed % 100 == 0:
            print(
                f"nodes={processed},open={len(heap)},global_upper="
                f"{-heap[0][0] if heap else target:.6f},pruned={pruned},depth={deepest}",
                flush=True,
            )

    global_upper = target if not heap else max(target, -heap[0][0])
    worst = None if not heap else heap[0][2]
    return {
        "parameterization": "ux_uy_explosion_delay",
        "dt": dt,
        "fine_dt": fine_dt,
        "refine_below": refine_below,
        "preset": preset,
        "cover_end": COVER_END,
        "distance_guard": DISTANCE_GUARD,
        "time_guard": TIME_GUARD,
        "n_phi_per_rim": n_phi,
        "center_cells": center_cells,
        "single_cumulative_cap": single_cumulative_cap,
        "consistent_center_cells": consistent_center_cells,
        "required_active_bombs": [i + 1 for i in required_active],
        "event_cells": event_cells,
        "joint_cell_consistency": joint_cell_consistency,
        "strong_branch_below": strong_branch_below,
        "explosion_order": [index + 1 for index in explosion_order],
        "theta_range_deg": None
        if theta_range is None
        else [math.degrees(value) for value in theta_range],
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
        else {
            "lo": worst.lo,
            "hi": worst.hi,
            "depth": worst.depth,
            "split_index": split_index(worst),
        },
    }


def self_test() -> None:
    theta_range = (math.radians(168.75), math.pi)
    points = legacy.witnesses(16)
    incumbent = strategy_box(q3.INCUMBENT)
    assert upper_duration(incumbent, 0.01, points) >= 7.650405706
    assert contract(incumbent, explosion_order=(0, 1)) is not None
    assert contract(incumbent, explosion_order=(0, 1, 2)) is not None
    assert contract(incumbent, explosion_order=(2, 1, 0)) is None
    paired = upper_duration(
        incumbent,
        0.01,
        points,
        center_cells=4,
        single_cumulative_cap=7.0,
        consistent_center_cells=True,
        event_cells=4,
        joint_cell_consistency=True,
    )
    assert paired >= 7.650405706

    # The interval cloud at a physical point must reproduce Strategy exactly.
    times = np.array([5.0, 9.0, 12.0])
    for strategy, (center, half_width, _) in zip(
        q3.INCUMBENT, cloud_boxes(incumbent, times, 0.0)
    ):
        expected = np.array([strategy.cloud_center(float(t)) for t in times])
        assert np.max(np.abs(center - expected)) < 5e-12
        assert np.max(np.abs(half_width)) < 5e-12

    # The hard heading shard must retain physical points from its annular sector.
    rng = np.random.default_rng(20260828)
    root = root_box("two", theta_range)
    for _ in range(100):
        theta = rng.uniform(*theta_range)
        speed = rng.uniform(SPEED_MIN, SPEED_MAX)
        releases = np.sort(rng.uniform([0.0, 1.0], [4.0, 6.0]))
        if releases[1] - releases[0] < 1.0:
            releases[1] = releases[0] + 1.0
        delays = rng.uniform(0.0, 4.0, size=2)
        values = (
            speed * math.cos(theta),
            speed * math.sin(theta),
            releases[0] + delays[0],
            releases[1] + delays[1],
            T,
            delays[0],
            delays[1],
            0.0,
        )
        point = contract(Box(values, values), (0, 1), theta_range)
        assert point is not None
        assert all(a - 1e-9 <= x <= b + 1e-9 for x, a, b in zip(values, root.lo, root.hi))

    # Ordering branches must cover every physical three-bomb point.
    for _ in range(100):
        releases = np.array(
            [
                rng.uniform(0.0, 2.0),
                rng.uniform(3.0, 4.0),
                rng.uniform(5.0, 6.0),
            ]
        )
        delays = np.array(
            [rng.uniform(0.0, min(T - release, 5.0)) for release in releases]
        )
        explosions = releases + delays
        order = tuple(int(index) for index in np.argsort(explosions))
        values = (
            -100.0,
            0.0,
            *explosions,
            *delays,
        )
        assert contract(
            Box(values, values),
            required_active=(0, 1, 2),
            explosion_order=order,
        ) is not None

    invalid = Box(
        (-100.0, 0.0, 2.0, 2.5, T, 0.0, 0.0, 0.0),
        (-100.0, 0.0, 2.0, 2.5, T, 0.0, 0.0, 0.0),
    )
    assert contract(invalid, theta_range=theta_range) is None

    inactive = Box(
        (-100.0, 0.0, T - 2.0, T - 1.0, T, 0.0, 0.0, 0.0),
        (-100.0, 0.0, T - 2.0, T - 1.0, T, 0.0, 0.0, 0.0),
    )
    assert upper_duration(
        inactive,
        0.05,
        points,
        center_cells=4,
        event_cells=4,
        joint_cell_consistency=True,
    ) == 0.0

    easy = branch_bound(
        0.05,
        None,
        10.0,
        16,
        COVER_END + 1e-9,
        0,
        1,
        "two",
        theta_range=theta_range,
    )
    assert easy["target_certified"] and easy["open"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fine-dt", type=float)
    parser.add_argument("--refine-below", type=float, default=11.0)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--target", type=float, default=8.0)
    parser.add_argument("--max-nodes", type=int, default=2_000)
    parser.add_argument("--center-cells", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--single-cumulative-cap", type=float)
    parser.add_argument("--consistent-center-cells", action="store_true")
    parser.add_argument("--required-active", default="")
    parser.add_argument("--event-cells", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--theta-range-deg")
    parser.add_argument("--joint-cell-consistency", action="store_true")
    parser.add_argument(
        "--strong-branch-below",
        type=float,
        help="try every non-degenerate split variable below this upper bound",
    )
    parser.add_argument(
        "--explosion-order",
        help="optional comma-separated 1-based nondecreasing explosion order",
    )
    parser.add_argument("--preset", choices=["full", "immediate", "two"], default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q3_upper_velocity_branch_bound.json"),
    )
    args = parser.parse_args()
    self_test()
    required_active = tuple(
        int(item) - 1 for item in args.required_active.split(",") if item
    )
    if any(index not in range(3) for index in required_active):
        parser.error("--required-active accepts only bomb numbers 1,2,3")
    theta_range = None
    if args.theta_range_deg:
        theta_values = [float(item) for item in args.theta_range_deg.split(",")]
        if len(theta_values) != 2 or not 0 <= theta_values[0] < theta_values[1] <= 360:
            parser.error("--theta-range-deg must be lower,upper within [0,360]")
        theta_range = tuple(math.radians(value) for value in theta_values)
    explosion_order = ()
    if args.explosion_order:
        explosion_order = tuple(
            int(item) - 1 for item in args.explosion_order.split(",") if item
        )
        if (
            len(explosion_order) < 2
            or len(set(explosion_order)) != len(explosion_order)
            or any(index not in range(3) for index in explosion_order)
        ):
            parser.error(
                "--explosion-order must list 2 or 3 distinct bomb numbers from 1,2,3"
            )
    result = branch_bound(
        args.dt,
        args.fine_dt,
        args.refine_below,
        args.n_phi,
        args.target,
        args.max_nodes,
        args.center_cells,
        args.preset,
        args.single_cumulative_cap,
        args.consistent_center_cells,
        required_active,
        args.event_cells,
        theta_range,
        args.joint_cell_consistency,
        args.strong_branch_below,
        explosion_order,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
