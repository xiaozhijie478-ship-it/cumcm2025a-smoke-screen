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
ROUTE_GUARD = 1e-4
ROUTE_TIME_EPS = 1e-3
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


def root_box(
    preset: str = "full",
    theta_range: tuple[float, float] | None = None,
) -> Box:
    theta_lo, theta_hi = theta_range or (0.0, 2 * math.pi)
    if preset == "immediate":
        return Box(
            (theta_lo, 140.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
            (theta_hi, 140.0, 0.0, T - 1.0, T, 0.0, TAU_MAX, TAU_MAX),
        )
    if preset == "two":
        # Any physical two-bomb strategy can be embedded by releasing an
        # inert third bomb at missile impact.  The release-gap contractor then
        # preserves the complete relevant two-bomb domain.
        return Box(
            (theta_lo, 70.0, 0.0, 1.0, T, 0.0, 0.0, 0.0),
            (theta_hi, 140.0, T - 2.0, T - 1.0, T, TAU_MAX, TAU_MAX, 0.0),
        )
    return Box(
        (theta_lo, 70.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0),
        (theta_hi, 140.0, T - 2.0, T - 1.0, T, TAU_MAX, TAU_MAX, TAU_MAX),
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


def contract(box: Box, required_active: tuple[int, ...] = ()) -> Box | None:
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
        for i in required_active:
            r, tau = 2 + i, 5 + i
            # A bomb that contributes before COVER_END must have exploded by
            # then.  Contract the box around r_i + tau_i <= COVER_END; the
            # remaining interval box is still an outer relaxation.
            hi[r] = min(hi[r], COVER_END - lo[tau])
            hi[tau] = min(hi[tau], COVER_END - lo[r])
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


def cloud_boxes(
    box: Box,
    times: np.ndarray,
    half_dt: float,
    required_active: tuple[int, ...] = (),
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return center(t), axis half-widths and possible-active masks."""
    lo, hi = box.lo, box.hi
    result = []
    for i in range(3):
        r = (lo[2 + i], hi[2 + i])
        tau = (lo[5 + i], hi[5 + i])
        te = (r[0] + tau[0], r[1] + tau[1])
        if i in required_active:
            te = (te[0], min(te[1], COVER_END))
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


def center_partition(widths: np.ndarray, center_cells: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return fixed offsets and half-widths for the adaptive center subboxes."""
    splits = np.ones(3, dtype=int)
    while int(np.prod(splits)) < center_cells and np.any(widths / splits > 1e-12):
        splits[int(np.argmax(widths / splits))] *= 2
    local_half = widths / splits
    return [
        (
            -widths + (2 * np.asarray(index) + 1) * local_half,
            local_half,
        )
        for index in np.ndindex(tuple(int(value) for value in splits))
    ]


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
    partitions = center_partition(widths, center_cells)
    count = len(partitions)
    masks = np.zeros((len(missiles), count), dtype=np.uint64)
    for cell, (offset, local_half) in enumerate(partitions):
        local_center = center + offset
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


def route_cells_compatible(
    box: Box,
    clouds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    partitions: list[list[tuple[np.ndarray, np.ndarray]]],
    event_edges: list[np.ndarray],
    choices: tuple[tuple[int, int], ...],
    relevant: list[int],
    time0: float,
) -> bool:
    """Necessary common-heading/common-speed test for paired cell choices."""
    speed_lo, speed_hi = box.lo[1], box.hi[1]
    theta_lo, theta_hi = box.lo[0], box.hi[0]
    theta_mid = (theta_lo + theta_hi) / 2
    use_angle = theta_hi - theta_lo < math.pi
    release_ranges: dict[int, list[float]] = {}

    for ball_index, (center_cell, event_cell) in zip(relevant, choices):
        offset, local_half = partitions[ball_index][center_cell]
        center_xy = clouds[ball_index][0][0, :2] + offset[:2] - q2.UAV_0[:2]
        rect_lo = center_xy - local_half[:2]
        rect_hi = center_xy + local_half[:2]

        gap = np.maximum(np.maximum(rect_lo, -rect_hi), 0.0)
        radius_lo = float(np.linalg.norm(gap))
        corners = np.array(list(itertools.product(*zip(rect_lo, rect_hi))))
        radius_hi = float(np.max(np.linalg.norm(corners, axis=1)))
        explosion_lo = float(event_edges[ball_index][event_cell])
        explosion_hi = float(event_edges[ball_index][event_cell + 1])

        vertical_center = (
            clouds[ball_index][0][0, 2] + offset[2] + 3.0 * time0
        )
        vertical_lo = vertical_center - local_half[2]
        vertical_hi = vertical_center + local_half[2]
        energy_lo = q2.UAV_0[2] + 3.0 * explosion_lo - vertical_hi
        energy_hi = q2.UAV_0[2] + 3.0 * explosion_hi - vertical_lo
        if energy_hi < -DISTANCE_GUARD:
            return False
        delay_lo = math.sqrt(max(0.0, 2.0 * energy_lo / q3.G))
        delay_hi = math.sqrt(max(0.0, 2.0 * energy_hi / q3.G))
        delay_lo = max(delay_lo, box.lo[5 + ball_index])
        delay_hi = min(delay_hi, box.hi[5 + ball_index])
        if delay_lo > delay_hi + ROUTE_GUARD:
            return False
        release_lo = max(box.lo[2 + ball_index], explosion_lo - delay_hi)
        release_hi = min(box.hi[2 + ball_index], explosion_hi - delay_lo)
        if release_lo > release_hi + ROUTE_GUARD:
            return False
        release_ranges[ball_index] = [release_lo, release_hi]

        # Near immediate explosion, subtracting the 17.8 km UAV coordinate to
        # recover a millimetre-scale horizontal displacement loses relative
        # precision.  Skip this optional route filter for that cell; keeping
        # it can only loosen the upper bound.
        if explosion_hi <= ROUTE_TIME_EPS:
            continue
        speed_lo = max(speed_lo, radius_lo / explosion_hi)
        if explosion_lo > ROUTE_TIME_EPS:
            speed_hi = min(speed_hi, radius_hi / explosion_lo)
        if speed_lo > speed_hi + ROUTE_GUARD:
            return False

        if use_angle and not (
            rect_lo[0] <= 0.0 <= rect_hi[0]
            and rect_lo[1] <= 0.0 <= rect_hi[1]
        ):
            angles = np.arctan2(corners[:, 1], corners[:, 0])
            angles = theta_mid + np.arctan2(
                np.sin(angles - theta_mid),
                np.cos(angles - theta_mid),
            )
            angle_lo, angle_hi = float(np.min(angles)), float(np.max(angles))
            if angle_hi - angle_lo <= math.pi:
                theta_lo = max(theta_lo, angle_lo)
                theta_hi = min(theta_hi, angle_hi)
                if theta_lo > theta_hi + ROUTE_GUARD:
                    return False

    ordered = sorted(release_ranges)
    for _ in range(len(ordered)):
        for left, right in zip(ordered, ordered[1:]):
            gap = float(right - left)
            release_ranges[right][0] = max(
                release_ranges[right][0], release_ranges[left][0] + gap
            )
            release_ranges[left][1] = min(
                release_ranges[left][1], release_ranges[right][1] - gap
            )
    return all(
        bounds[0] <= bounds[1] + ROUTE_GUARD
        for bounds in release_ranges.values()
    )


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
) -> float:
    """Safe duration upper bound for all strategies in ``box``."""
    contracted = contract(box, required_active)
    if contracted is None:
        return -math.inf
    half = dt / 2
    times = np.arange(half, COVER_END, dt)
    widths = np.minimum(times + half, COVER_END) - np.maximum(times - half, 0.0)
    missiles = geometry.MISSILE_0 + geometry.MISSILE_SPEED * times[:, None] * geometry.MISSILE_DIRECTION
    clouds = cloud_boxes(contracted, times, half, required_active)
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

    consistent_upper: float | None = None
    masks: list[np.ndarray] | None = None
    if (center_cells > 1 or event_cells > 1) and len(points) <= 63 and np.any(possible):
        masks = [
            coverage_masks(center, half_width, active, missiles, points, threshold, center_cells)
            for center, half_width, active in clouds
        ]
        full = (np.uint64(1) << np.uint64(len(points))) - np.uint64(1)
        if consistent_center_cells:
            # A fixed physical strategy gives each bomb one time-independent
            # horizontal center and one time-independent vertical offset; the
            # cloud then only translates downward by 3t.  It must therefore
            # remain in one corresponding center subbox for the whole time
            # axis.  Enumerating one fixed subbox per bomb is still an outer
            # relaxation inside each subbox, but avoids changing subboxes in
            # every time bin.
            consistent_upper = 0.0
            for cells in itertools.product(*(range(item.shape[1]) for item in masks)):
                combined = masks[0][:, cells[0]].copy()
                for ball_index in range(1, len(masks)):
                    combined |= masks[ball_index][:, cells[ball_index]]
                feasible = possible & (combined == full)
                consistent_upper = max(
                    consistent_upper,
                    float(np.sum(widths[feasible])),
                )
        elif center_cells > 1:
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
    upper = (
        consistent_upper
        if consistent_upper is not None
        else float(np.sum(widths[possible]))
    )
    if event_cells > 1 and masks is not None:
        # A fixed bomb has one explosion time, hence one 20 s active interval.
        # The ordinary interval mask uses the union over every explosion time
        # in the parameter box and can implicitly move that interval between
        # time bins.  Split each explosion-time range, choose one cell per bomb
        # for the whole axis, and keep the most optimistic combination.
        event_masks: list[list[np.ndarray]] = []
        event_active_masks: list[list[np.ndarray]] = []
        event_edges: list[np.ndarray] = []
        for i, mask in enumerate(masks):
            lo_e = contracted.lo[2 + i] + contracted.lo[5 + i]
            hi_e = contracted.hi[2 + i] + contracted.hi[5 + i]
            if i in required_active:
                hi_e = min(hi_e, COVER_END)
            edges = np.linspace(lo_e, hi_e, event_cells + 1)
            event_edges.append(edges)
            any_center = np.bitwise_or.reduce(mask, axis=1)
            choices = []
            active_choices = []
            for cell in range(event_cells):
                active = (edges[cell] <= times + half + TIME_GUARD) & (
                    edges[cell + 1] + geometry.SMOKE_LIFETIME
                    >= times - half - TIME_GUARD
                )
                active_choices.append(active)
                choices.append(np.where(active, any_center, np.uint64(0)))
            event_masks.append(choices)
            event_active_masks.append(active_choices)
        event_upper = 0.0
        for choices in itertools.product(range(event_cells), repeat=len(masks)):
            combined = event_masks[0][choices[0]].copy()
            for ball_index in range(1, len(event_masks)):
                combined |= event_masks[ball_index][choices[ball_index]]
            feasible = combined == full
            event_upper = max(event_upper, float(np.sum(widths[feasible])))
        upper = min(upper, event_upper)
        if joint_cell_consistency:
            # A physical bomb simultaneously belongs to one center subbox and
            # one explosion-time subcell.  Enumerate those paired choices for
            # the entire time axis instead of optimizing the two partitions
            # separately.  Geometry inside each paired cell is still relaxed.
            relevant = [index for index, mask in enumerate(masks) if np.any(mask)]
            if not relevant:
                return 0.0
            partitions = [
                center_partition(half_width[0], center_cells)
                for _, half_width, _ in clouds
            ]
            paired_choices = [
                list(itertools.product(range(masks[index].shape[1]), range(event_cells)))
                for index in relevant
            ]
            paired_upper = 0.0
            for choices in itertools.product(*paired_choices):
                if not route_cells_compatible(
                    contracted,
                    clouds,
                    partitions,
                    event_edges,
                    choices,
                    relevant,
                    float(times[0]),
                ):
                    continue
                center_cell, event_cell = choices[0]
                first = relevant[0]
                combined = np.where(
                    event_active_masks[first][event_cell],
                    masks[first][:, center_cell],
                    np.uint64(0),
                )
                for choice_index in range(1, len(relevant)):
                    ball_index = relevant[choice_index]
                    center_cell, event_cell = choices[choice_index]
                    combined |= np.where(
                        event_active_masks[ball_index][event_cell],
                        masks[ball_index][:, center_cell],
                        np.uint64(0),
                    )
                feasible = combined == full
                paired_upper = max(
                    paired_upper,
                    float(np.sum(widths[feasible])),
                )
            upper = min(upper, paired_upper)
    if single_cumulative_cap is not None:
        # An independently certified one-ball cumulative upper remains valid
        # for each bomb.  This matters after branching has proved that only a
        # subset of the three bombs can be active before COVER_END.
        possible_balls = sum(bool(np.any(active)) for _, _, active in clouds)
        upper = min(upper, possible_balls * single_cumulative_cap)
    return upper


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
    single_cumulative_cap: float | None = None,
    consistent_center_cells: bool = False,
    required_active: tuple[int, ...] = (),
    event_cells: int = 1,
    theta_range: tuple[float, float] | None = None,
    joint_cell_consistency: bool = False,
) -> dict[str, object]:
    points = witnesses(n_phi)
    root = contract(root_box(preset, theta_range), required_active)
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
                ),
            )
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
            child = contract(child, required_active)
            if child is None:
                infeasible += 1
                continue
            # A child box is a subset of its parent, so the parent's safe
            # upper bound remains valid for it.  Keeping that inherited cap
            # prevents harmless interval/subbox discretization changes from
            # making the reported global bound rise after a split.
            child_upper = min(upper, bound(child))
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
        "route_guard": ROUTE_GUARD,
        "route_time_epsilon": ROUTE_TIME_EPS,
        "n_phi_per_rim": n_phi,
        "center_cells": center_cells,
        "single_cumulative_cap": single_cumulative_cap,
        "consistent_center_cells": consistent_center_cells,
        "required_active_bombs": [index + 1 for index in required_active],
        "event_cells": event_cells,
        "joint_cell_consistency": joint_cell_consistency,
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
        else {"lo": worst.lo, "hi": worst.hi, "depth": worst.depth, "split_index": split_index(worst)},
    }


def self_test() -> None:
    points = witnesses(16)
    incumbent = point_box()
    upper = upper_duration(incumbent, 0.01, points)
    assert upper >= 7.650405706
    consistent = upper_duration(
        incumbent,
        0.01,
        points,
        center_cells=4,
        consistent_center_cells=True,
    )
    assert 7.650405706 <= consistent <= upper + 1e-12
    event_consistent = upper_duration(
        incumbent,
        0.01,
        points,
        center_cells=4,
        event_cells=4,
    )
    assert 7.650405706 <= event_consistent <= upper + 1e-12
    paired_consistent = upper_duration(
        incumbent,
        0.01,
        points,
        center_cells=4,
        event_cells=4,
        joint_cell_consistency=True,
    )
    assert 7.650405706 <= paired_consistent <= event_consistent + 1e-12
    # Regression for large-coordinate cancellation when a bomb explodes
    # almost immediately after release.
    near_immediate_values = (
        math.radians(182.05952238601665),
        140.0,
        8.77768040631848e-10,
        2.427451017329057,
        4.983345971320381,
        3.737831996772812e-06,
        4.97088495566696,
        5.8476180845065615,
    )
    near_immediate = Box(near_immediate_values, near_immediate_values)
    guarded = upper_duration(
        near_immediate,
        0.02,
        points,
        center_cells=4,
        event_cells=4,
        joint_cell_consistency=True,
    )
    assert guarded >= 2.55
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
    parser.add_argument(
        "--single-cumulative-cap",
        type=float,
        help="optional independently certified cumulative upper for one bomb",
    )
    parser.add_argument(
        "--consistent-center-cells",
        action="store_true",
        help="keep one spatial subbox per bomb across all time bins",
    )
    parser.add_argument(
        "--required-active",
        default="",
        help="comma-separated 1-based bombs required to explode by COVER_END",
    )
    parser.add_argument(
        "--event-cells",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="fixed explosion-time subcells per bomb across all time bins",
    )
    parser.add_argument(
        "--theta-range-deg",
        help="optional lower,upper heading shard in degrees",
    )
    parser.add_argument(
        "--joint-cell-consistency",
        action="store_true",
        help="pair fixed center and explosion-time cells across the time axis",
    )
    parser.add_argument("--preset", choices=["full", "immediate", "two"], default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q3_upper_branch_bound.json"),
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
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
