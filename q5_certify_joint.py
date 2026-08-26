"""Certified continuous-surface/time audit for the Q5 joint smoke plan."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
from pathlib import Path

import numpy as np

import q1_strict_occlusion as geometry
import q4_optimize as q4
import q5_optimize as q5


TIME_LIPSCHITZ = geometry.MISSILE_SPEED + geometry.SMOKE_SINK_SPEED
# Double precision is used rather than outward-rounded interval arithmetic.
# Keep a small one-sided guard in every proof decision so values that land on
# the analytical threshold are left unresolved instead of being overclaimed.
NUMERICAL_GUARD = 1e-8


def missile_position(name: str, t: float) -> np.ndarray:
    start = q5.MISSILES[name]
    return start - geometry.MISSILE_SPEED * t * start / np.linalg.norm(start)


def point_value(
    point: np.ndarray,
    missile: np.ndarray,
    clouds: np.ndarray,
) -> float:
    segment = point - missile
    denominator = float(np.dot(segment, segment))
    fractions = np.clip(((clouds - missile) @ segment) / denominator, 0.0, 1.0)
    closest = missile + fractions[:, None] * segment
    return float(np.min(np.linalg.norm(closest - clouds, axis=1)))


def point_distances(
    point: np.ndarray,
    missile: np.ndarray,
    clouds: np.ndarray,
) -> np.ndarray:
    segment = point - missile
    denominator = float(np.dot(segment, segment))
    fractions = np.clip(((clouds - missile) @ segment) / denominator, 0.0, 1.0)
    closest = missile + fractions[:, None] * segment
    return np.linalg.norm(closest - clouds, axis=1)


def quadratic_minimum(a: float, b: float, c: float, left: float, right: float) -> float:
    values = [a * left**2 + b * left + c, a * right**2 + b * right + c]
    if abs(a) > 1e-15:
        vertex = -b / (2 * a)
        if left < vertex < right:
            values.append(a * vertex**2 + b * vertex + c)
    return min(values)


def point_time_lipschitz(
    point: np.ndarray,
    missile_name: str,
    strategy: q5.BombStrategy,
    left: float,
    right: float,
) -> float:
    """Bound time variation at one target point over a fixed-active-set interval."""
    missile_0 = q5.MISSILES[missile_name]
    missile_velocity = -geometry.MISSILE_SPEED * missile_0 / np.linalg.norm(missile_0)
    cloud_velocity = np.array([0.0, 0.0, -geometry.SMOKE_SINK_SPEED])
    cloud_0 = strategy.explosion_point - cloud_velocity * strategy.explosion_time
    q_0 = cloud_0 - missile_0
    q_velocity = cloud_velocity - missile_velocity
    segment_0 = point - missile_0
    # N(t)=(C-M) dot (P-M). N>0 excludes the fast-moving missile endpoint.
    coefficient_2 = -float(np.dot(q_velocity, missile_velocity))
    coefficient_1 = float(np.dot(q_velocity, segment_0) - np.dot(q_0, missile_velocity))
    coefficient_0 = float(np.dot(q_0, segment_0))
    numerator_min = quadratic_minimum(
        coefficient_2, coefficient_1, coefficient_0, left, right
    )
    if numerator_min <= 0.0:
        return TIME_LIPSCHITZ
    closest_time = float(
        np.clip(
            np.dot(segment_0, missile_velocity) / np.dot(missile_velocity, missile_velocity),
            left,
            right,
        )
    )
    minimum_segment_length = float(
        np.linalg.norm(segment_0 - missile_velocity * closest_time)
    )
    perpendicular_speed = float(
        np.linalg.norm(np.cross(missile_velocity, segment_0)) / minimum_segment_length
    )
    return geometry.SMOKE_SINK_SPEED + perpendicular_speed


def cell_point_radius(cell: tuple[str, float, float, float, float]) -> tuple[np.ndarray, float]:
    kind, phi_0, phi_1, first_0, first_1 = cell
    phi = (phi_0 + phi_1) / 2
    half_phi = (phi_1 - phi_0) / 2
    if kind == "side":
        z = (first_0 + first_1) / 2
        point = np.array(
            [
                geometry.TARGET_RADIUS * math.cos(phi),
                geometry.TARGET_CENTER_XY[1] + geometry.TARGET_RADIUS * math.sin(phi),
                z,
            ]
        )
        angular = 2 * geometry.TARGET_RADIUS * math.sin(half_phi / 2)
        radius = math.hypot(angular, (first_1 - first_0) / 2)
        return point, radius

    radial = (first_0 + first_1) / 2
    z = 0.0 if kind == "bottom" else geometry.TARGET_HEIGHT
    point = np.array(
        [
            radial * math.cos(phi),
            geometry.TARGET_CENTER_XY[1] + radial * math.sin(phi),
            z,
        ]
    )
    cosine = math.cos(half_phi)
    radius = max(
        math.sqrt(max(0.0, edge**2 + radial**2 - 2 * edge * radial * cosine))
        for edge in (first_0, first_1)
    )
    return point, radius


def split_cell(
    cell: tuple[str, float, float, float, float]
) -> tuple[tuple[str, float, float, float, float], tuple[str, float, float, float, float]]:
    kind, phi_0, phi_1, first_0, first_1 = cell
    half_phi = (phi_1 - phi_0) / 2
    if kind == "side":
        angular = 2 * geometry.TARGET_RADIUS * math.sin(half_phi / 2)
        split_phi = angular >= (first_1 - first_0) / 2
    else:
        angular = 2 * first_1 * math.sin(half_phi / 2)
        split_phi = angular >= (first_1 - first_0) / 2
    if split_phi:
        midpoint = (phi_0 + phi_1) / 2
        return (
            (kind, phi_0, midpoint, first_0, first_1),
            (kind, midpoint, phi_1, first_0, first_1),
        )
    midpoint = (first_0 + first_1) / 2
    return (
        (kind, phi_0, phi_1, first_0, midpoint),
        (kind, phi_0, phi_1, midpoint, first_1),
    )


def initial_cells(n_phi: int = 12, n_linear: int = 3):
    for phi_index in range(n_phi):
        phi_0 = 2 * math.pi * phi_index / n_phi
        phi_1 = 2 * math.pi * (phi_index + 1) / n_phi
        for index in range(n_linear):
            yield (
                "side",
                phi_0,
                phi_1,
                geometry.TARGET_HEIGHT * index / n_linear,
                geometry.TARGET_HEIGHT * (index + 1) / n_linear,
            )
            for kind in ("bottom", "top"):
                yield (
                    kind,
                    phi_0,
                    phi_1,
                    geometry.TARGET_RADIUS * index / n_linear,
                    geometry.TARGET_RADIUS * (index + 1) / n_linear,
                )


def surface_bounds(
    missile_name: str,
    t: float,
    strategies: list[q5.BombStrategy],
    tolerance: float,
    max_cells: int,
    time_half_width: float | None = None,
) -> tuple[float, float, int]:
    active = [
        item
        for item in strategies
        if item.explosion_time < t < item.explosion_time + geometry.SMOKE_LIFETIME
    ]
    if not active:
        return math.inf, math.inf, 0
    missile = missile_position(missile_name, t)
    clouds = np.stack([item.cloud_center(t) for item in active])
    heap: list[tuple[float, int, tuple[str, float, float, float, float]]] = []
    counter = itertools.count()
    lower = 0.0

    def add(cell: tuple[str, float, float, float, float]) -> None:
        nonlocal lower
        point, radius = cell_point_radius(cell)
        value = point_value(point, missile, clouds)
        lower = max(lower, value)
        heapq.heappush(heap, (-(value + radius), next(counter), cell))

    for cell in initial_cells():
        add(cell)
    def classification_decided() -> bool:
        if time_half_width is None:
            return False
        upper = -heap[0][0]
        cover_threshold = geometry.SMOKE_RADIUS - TIME_LIPSCHITZ * time_half_width
        uncover_threshold = geometry.SMOKE_RADIUS + TIME_LIPSCHITZ * time_half_width
        return (
            upper <= cover_threshold
            or lower > uncover_threshold
            or (lower > cover_threshold and upper <= uncover_threshold)
        )

    while (
        -heap[0][0] - lower > tolerance
        and len(heap) < max_cells
        and not classification_decided()
    ):
        _, _, cell = heapq.heappop(heap)
        for child in split_cell(cell):
            add(child)
    return lower, -heap[0][0], len(heap)


def interval_surface_status(
    missile_name: str,
    left: float,
    right: float,
    strategies: list[q5.BombStrategy],
    max_cells: int,
) -> tuple[str, int, float]:
    """Prove a whole time-surface block covered/uncovered, or return ambiguous."""
    midpoint = (left + right) / 2
    half_width = (right - left) / 2

    # A proof block may not cross a smoke activation or expiry event.  Using
    # the midpoint active set alone would otherwise let a ball certify times
    # at which it does not yet exist (or has already expired).
    for item in strategies:
        for event in (item.explosion_time, item.explosion_time + geometry.SMOKE_LIFETIME):
            if left + NUMERICAL_GUARD < event < right - NUMERICAL_GUARD:
                return "ambiguous", 0, math.inf
    active = [
        item
        for item in strategies
        if item.explosion_time <= left + NUMERICAL_GUARD
        and right <= item.explosion_time + geometry.SMOKE_LIFETIME + NUMERICAL_GUARD
    ]
    if not active:
        return "uncovered", 0, math.inf
    missile = missile_position(missile_name, midpoint)
    clouds = np.stack([item.cloud_center(midpoint) for item in active])
    heap: list[tuple[float, int, tuple[str, float, float, float, float]]] = []
    counter = itertools.count()
    cells = 0
    worst_upper = 0.0

    def add(cell: tuple[str, float, float, float, float]) -> bool:
        nonlocal cells, worst_upper
        point, radius = cell_point_radius(cell)
        distances = point_distances(point, missile, clouds)
        rates = np.array(
            [
                point_time_lipschitz(point, missile_name, item, left, right)
                for item in active
            ]
        )
        witness_lower = float(np.min(distances - rates * half_width))
        if witness_lower > geometry.SMOKE_RADIUS + NUMERICAL_GUARD:
            return True
        cover_upper = float(np.min(distances + rates * half_width)) + radius
        worst_upper = max(worst_upper, cover_upper)
        heapq.heappush(heap, (-cover_upper, next(counter), cell))
        cells += 1
        return False

    for cell in initial_cells():
        if add(cell):
            return "uncovered", cells, worst_upper
    cover_threshold = geometry.SMOKE_RADIUS - NUMERICAL_GUARD
    while -heap[0][0] > cover_threshold and cells < max_cells:
        _, _, cell = heapq.heappop(heap)
        for child in split_cell(cell):
            if add(child):
                return "uncovered", cells, worst_upper
    return (
        "covered" if -heap[0][0] <= cover_threshold else "ambiguous",
        cells,
        -heap[0][0],
    )


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return geometry.merge(intervals, tol=1e-12)


def single_observer(
    strategy: q5.BombStrategy, missile_name: str, t: float
) -> float:
    missile = missile_position(missile_name, t)
    cloud = strategy.cloud_center(t)
    return max(
        geometry.ring_max_violation(missile, cloud, z)[0]
        for z in (0.0, geometry.TARGET_HEIGHT)
    )


def single_boundary(
    strategy: q5.BombStrategy,
    missile_name: str,
    left: float,
    right: float,
) -> float:
    f_left = single_observer(strategy, missile_name, left)
    for _ in range(60):
        midpoint = (left + right) / 2
        f_mid = single_observer(strategy, missile_name, midpoint)
        if f_left * f_mid <= 0:
            right = midpoint
        else:
            left, f_left = midpoint, f_mid
    return (left + right) / 2


def single_intervals(
    strategy: q5.BombStrategy,
    missile_name: str,
    step: float = 0.02,
) -> list[tuple[float, float]]:
    start = strategy.explosion_time
    end = min(
        strategy.explosion_time + geometry.SMOKE_LIFETIME,
        q5.missile_hit_time(missile_name),
    )


def local_single_intervals(
    strategy: q5.BombStrategy,
    missile_name: str,
    candidates: list[list[float]],
    step: float = 0.02,
    padding: float = 0.05,
) -> list[tuple[float, float]]:
    """Refine known dense-grid intervals; omitted intervals only weaken the lower bound."""
    active_start = strategy.explosion_time
    active_end = min(
        strategy.explosion_time + geometry.SMOKE_LIFETIME,
        q5.missile_hit_time(missile_name),
    )
    refined = []
    for candidate_start, candidate_end in candidates:
        start = max(active_start, float(candidate_start) - padding)
        end = min(active_end, float(candidate_end) + padding)
        if end <= start:
            continue
        times = np.linspace(start, end, max(2, int(math.ceil((end - start) / step)) + 1))
        values = np.array(
            [single_observer(strategy, missile_name, float(t)) for t in times]
        )
        roots = []
        for index in range(len(times) - 1):
            if values[index] == 0:
                roots.append(float(times[index]))
            elif values[index] * values[index + 1] < 0:
                roots.append(
                    single_boundary(
                        strategy,
                        missile_name,
                        float(times[index]),
                        float(times[index + 1]),
                    )
                )
        cuts = [start, *roots, end]
        refined.extend(
            (left, right)
            for left, right in zip(cuts, cuts[1:])
            if right > left
            and single_observer(strategy, missile_name, (left + right) / 2) <= 0
        )
    return merge(refined)
    if end <= start:
        return []
    times = np.linspace(start, end, max(2, int(math.ceil((end - start) / step)) + 1))
    values = np.array([single_observer(strategy, missile_name, float(t)) for t in times])
    roots = []
    for index in range(len(times) - 1):
        if values[index] == 0:
            roots.append(float(times[index]))
        elif values[index] * values[index + 1] < 0:
            roots.append(
                single_boundary(
                    strategy, missile_name, float(times[index]), float(times[index + 1])
                )
            )
    cuts = [start, *roots, end]
    return merge(
        [
            (left, right)
            for left, right in zip(cuts, cuts[1:])
            if right > left
            and single_observer(strategy, missile_name, (left + right) / 2) <= 0
        ]
    )


def interval_difference(
    whole: list[tuple[float, float]],
    subtract: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result = []
    for start, end in whole:
        cursor = start
        for left, right in subtract:
            if right <= cursor or left >= end:
                continue
            if left > cursor:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def certify_candidate_intervals(
    missile_name: str,
    candidates: list[tuple[float, float]],
    strategies: list[q5.BombStrategy],
    time_tolerance: float,
    max_cells: int,
    initial_chunk: float = 0.005,
) -> dict[str, object]:
    events = sorted(
        {
            value
            for item in strategies
            for value in (
                item.explosion_time,
                item.explosion_time + geometry.SMOKE_LIFETIME,
            )
        }
    )
    stack = []
    for start, end in candidates:
        cuts = [start, *(value for value in events if start < value < end), end]
        for left, right in zip(cuts, cuts[1:]):
            count = max(1, int(math.ceil((right - left) / initial_chunk)))
            local = np.linspace(left, right, count + 1)
            stack.extend((float(a), float(b)) for a, b in zip(local, local[1:]))
    covered: list[tuple[float, float]] = []
    unresolved: list[tuple[float, float]] = []
    rejected: list[tuple[float, float]] = []
    nodes = peak_cells = 0
    while stack:
        left, right = stack.pop()
        # Broad boundary blocks are cheap to split in time and expensive to
        # over-refine in space.  Reserve the full surface-cell budget for the
        # final narrow blocks where it can actually settle the classification.
        local_max_cells = (
            max_cells
            if right - left <= 2.0 * time_tolerance
            else min(max_cells, 3_000)
        )
        status, cells, _ = interval_surface_status(
            missile_name, left, right, strategies, local_max_cells
        )
        nodes += 1
        peak_cells = max(peak_cells, cells)
        if status == "covered":
            covered.append((left, right))
        elif status == "uncovered":
            rejected.append((left, right))
        elif right - left <= time_tolerance:
            unresolved.append((left, right))
        else:
            midpoint = (left + right) / 2
            stack.extend(((left, midpoint), (midpoint, right)))
        if nodes % 100 == 0:
            print(
                f"hybrid_missile={missile_name},nodes={nodes},stack={len(stack)},"
                f"covered={sum(b-a for a,b in covered):.6f}",
                flush=True,
            )
    covered, unresolved, rejected = map(merge, (covered, unresolved, rejected))
    return {
        "covered": covered,
        "unresolved": unresolved,
        "rejected": rejected,
        "duration_lower": sum(end - start for start, end in covered),
        "duration_unresolved": sum(end - start for start, end in unresolved),
        "nodes": nodes,
        "peak_surface_cells": peak_cells,
    }


def hybrid_certificate(
    validation_file: Path,
    bomb_table_file: Path,
    strategies: list[q5.BombStrategy],
    time_tolerance: float,
    max_cells: int,
) -> dict[str, object]:
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    bomb_table = json.loads(bomb_table_file.read_text(encoding="utf-8"))["bombs"]
    approximate = {
        (item["uav"], int(item["number"])): item["individual_intervals"]
        for item in bomb_table
    }
    dense = validation["resolutions"]["dense"]["metrics"]["intervals"]
    by_missile = {}
    for missile_name in q5.MISSILES:
        individual = {}
        for item in strategies:
            intervals = (
                local_single_intervals(
                    item,
                    missile_name,
                    approximate[(item.uav, item.number)],
                )
                if item.assigned_missile == missile_name
                else []
            )
            individual[f"{item.uav}-{item.number}"] = intervals
            if intervals:
                print(
                    f"single_exact={missile_name},{item.uav}-{item.number},"
                    f"duration={sum(b-a for a,b in intervals):.9f}",
                    flush=True,
                )
        single_union = merge([pair for intervals in individual.values() for pair in intervals])
        joint_only = interval_difference(
            [tuple(pair) for pair in dense[missile_name]], single_union
        )
        joint_certificate = certify_candidate_intervals(
            missile_name, joint_only, strategies, time_tolerance, max_cells
        )
        single_duration = sum(end - start for start, end in single_union)
        by_missile[missile_name] = {
            "individual_intervals": individual,
            "single_baseline_scope": "assigned bombs only; omitted cross-label effects can only raise the lower bound",
            "single_union": single_union,
            "single_duration": single_duration,
            "dense_joint_only_candidates": joint_only,
            "joint_only_certificate": joint_certificate,
            "certified_duration_lower": single_duration
            + float(joint_certificate["duration_lower"]),
            "dense_duration": sum(end - start for start, end in dense[missile_name]),
        }
        print(
            f"hybrid_done={missile_name},single={single_duration:.9f},"
            f"joint_certified={joint_certificate['duration_lower']:.9f},"
            f"dense={by_missile[missile_name]['dense_duration']:.9f}",
            flush=True,
        )
    return {
        "by_missile": by_missile,
        "total_certified_lower": sum(
            float(item["certified_duration_lower"]) for item in by_missile.values()
        ),
        "total_dense": sum(float(item["dense_duration"]) for item in by_missile.values()),
    }


def targeted_certificate(
    validation_file: Path,
    bomb_table_file: Path,
    strategies: list[q5.BombStrategy],
    max_cells: int,
) -> dict[str, object]:
    """Certify representative, disjoint pieces of every material joint-only interval."""
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    dense = validation["resolutions"]["dense"]["metrics"]["intervals"]
    bomb_table = json.loads(bomb_table_file.read_text(encoding="utf-8"))["bombs"]
    approximate = {
        (item["uav"], int(item["number"])): item["individual_intervals"]
        for item in bomb_table
    }
    results = {}
    for missile_name in q5.MISSILES:
        exact_individual = []
        for item in strategies:
            if item.assigned_missile != missile_name:
                continue
            intervals = local_single_intervals(
                item, missile_name, approximate[(item.uav, item.number)]
            )
            exact_individual.extend(intervals)
            print(
                f"targeted_single={missile_name},{item.uav}-{item.number},"
                f"duration={sum(b-a for a,b in intervals):.9f}",
                flush=True,
            )
        single_union = merge(exact_individual)
        dense_intervals = [tuple(pair) for pair in dense[missile_name]]
        joint_only = [
            pair
            for pair in interval_difference(dense_intervals, single_union)
            if pair[1] - pair[0] >= 0.05
        ]
        chunk_width = 0.005 if missile_name == "M3" else 0.02
        certified = []
        attempts = []
        for start, end in joint_only:
            centers = np.arange(start + 0.025, end, 0.05)
            if len(centers) == 0:
                centers = np.array([(start + end) / 2])
            for center in centers:
                half = min(chunk_width / 2, center - start, end - center)
                left, right = float(center - half), float(center + half)
                status, cells, upper = interval_surface_status(
                    missile_name, left, right, strategies, max_cells
                )
                attempts.append(
                    {
                        "interval": [left, right],
                        "status": status,
                        "cells": cells,
                        "upper": upper,
                    }
                )
                if status == "covered":
                    certified.append((left, right))
                print(
                    f"targeted_joint={missile_name},interval=({left:.6f},{right:.6f}),"
                    f"status={status},cells={cells},upper={upper:.6f}",
                    flush=True,
                )
        certified = merge(certified)
        single_duration = sum(end - start for start, end in single_union)
        joint_duration = sum(end - start for start, end in certified)
        results[missile_name] = {
            "single_union": single_union,
            "single_duration": single_duration,
            "material_joint_only_intervals": joint_only,
            "attempts": attempts,
            "certified_joint_pieces": certified,
            "certified_joint_duration": joint_duration,
            "certified_duration_lower": single_duration + joint_duration,
            "dense_duration": sum(end - start for start, end in dense_intervals),
        }
    return {
        "by_missile": results,
        "total_certified_lower": sum(
            float(item["certified_duration_lower"]) for item in results.values()
        ),
        "total_dense": sum(float(item["dense_duration"]) for item in results.values()),
    }


def outer_certificate(
    missile_name: str,
    strategies: list[q5.BombStrategy],
    hinted: list[tuple[float, float]],
    time_tolerance: float,
    max_cells: int,
) -> dict[str, object]:
    """Upper-bound the full continuous-time coverage set.

    Dense-grid intervals are used only as *possible* coverage regions and are
    therefore included wholesale in the upper bound.  Every part of the time
    axis outside those hints is independently proved uncovered or retained as
    unresolved.  Consequently a missed dense-grid interval cannot invalidate
    the upper bound; it merely appears in the outside unresolved/covered set.
    """
    hit = q5.missile_hit_time(missile_name)
    hints = merge(
        [(max(0.0, a), min(hit, b)) for a, b in hinted if b > 0.0 and a < hit]
    )
    events = {0.0, hit}
    for item in strategies:
        for value in (item.explosion_time, item.explosion_time + geometry.SMOKE_LIFETIME):
            if 0.0 < value < hit:
                events.add(value)
    for left, right in hints:
        events.update((left, right))
    cuts = sorted(events)

    possible: list[tuple[float, float]] = list(hints)
    outside_covered: list[tuple[float, float]] = []
    outside_unresolved: list[tuple[float, float]] = []
    outside_rejected: list[tuple[float, float]] = []
    stack: list[tuple[float, float]] = []
    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left + right) / 2
        if any(a <= midpoint <= b for a, b in hints):
            continue
        stack.append((left, right))

    nodes = peak_cells = 0
    while stack:
        left, right = stack.pop()
        local_max_cells = (
            max_cells
            if right - left <= 2.0 * time_tolerance
            else min(max_cells, 3_000)
        )
        status, cells, _ = interval_surface_status(
            missile_name, left, right, strategies, local_max_cells
        )
        nodes += 1
        peak_cells = max(peak_cells, cells)
        if status == "covered":
            outside_covered.append((left, right))
        elif status == "uncovered":
            outside_rejected.append((left, right))
        elif right - left <= time_tolerance:
            outside_unresolved.append((left, right))
        else:
            midpoint = (left + right) / 2
            stack.extend(((left, midpoint), (midpoint, right)))
        if nodes % 100 == 0:
            print(
                f"outer_missile={missile_name},nodes={nodes},stack={len(stack)},"
                f"outside_unresolved={sum(b-a for a,b in outside_unresolved):.6f}",
                flush=True,
            )

    possible = merge(possible + outside_covered + outside_unresolved)
    return {
        "missile": missile_name,
        "hinted_possible": hints,
        "outside_covered": merge(outside_covered),
        "outside_unresolved": merge(outside_unresolved),
        "outside_rejected": merge(outside_rejected),
        "duration_upper": sum(b - a for a, b in possible),
        "possible_intervals": possible,
        "nodes": nodes,
        "peak_surface_cells": peak_cells,
    }


def certify_missile(
    missile_name: str,
    strategies: list[q5.BombStrategy],
    space_tolerance: float,
    time_tolerance: float,
    max_cells: int,
) -> dict[str, object]:
    hit = q5.missile_hit_time(missile_name)
    events = {0.0, hit}
    for item in strategies:
        if 0.0 < item.explosion_time < hit:
            events.add(item.explosion_time)
        expiry = item.explosion_time + geometry.SMOKE_LIFETIME
        if 0.0 < expiry < hit:
            events.add(expiry)
    cuts = sorted(events)
    covered: list[tuple[float, float]] = []
    unresolved: list[tuple[float, float]] = []
    nodes = surface_calls = peak_cells = 0
    stack = [(left, right) for left, right in zip(cuts, cuts[1:])]
    while stack:
        left, right = stack.pop()
        midpoint = (left + right) / 2
        local_max_cells = (
            max_cells
            if right - left <= 2.0 * time_tolerance
            else min(max_cells, 3_000)
        )
        status, cells, _ = interval_surface_status(
            missile_name, left, right, strategies, local_max_cells
        )
        nodes += 1
        surface_calls += 1
        peak_cells = max(peak_cells, cells)
        if status == "covered":
            covered.append((left, right))
        elif status == "uncovered":
            pass
        elif right - left <= time_tolerance:
            unresolved.append((left, right))
        else:
            stack.extend(((left, midpoint), (midpoint, right)))
        if nodes % 100 == 0:
            print(
                f"missile={missile_name},nodes={nodes},stack={len(stack)},"
                f"covered={sum(b-a for a,b in covered):.6f},unresolved={sum(b-a for a,b in unresolved):.6f}",
                flush=True,
            )

    covered = merge(covered)
    unresolved = merge(unresolved)
    lower_duration = sum(right - left for left, right in covered)
    unresolved_duration = sum(right - left for left, right in unresolved)
    return {
        "missile": missile_name,
        "covered": covered,
        "unresolved": unresolved,
        "duration_lower": lower_duration,
        "duration_upper": lower_duration + unresolved_duration,
        "nodes": nodes,
        "surface_calls": surface_calls,
        "peak_surface_cells": peak_cells,
    }


def self_test(strategies: list[q5.BombStrategy]) -> None:
    # The cell bounds must enclose an independent dense-grid lower bound.
    for missile_name, t in (("M1", 15.0), ("M2", 17.0), ("M3", 18.0)):
        lower, upper, _ = surface_bounds(missile_name, t, strategies, 0.05, 20_000)
        dense = q5.joint_worst_distances(
            strategies,
            missile_name,
            np.array([t]),
            q4.surface_points(192, 41, 28),
        )[0]
        assert lower <= dense + 1e-10 <= upper + 1e-10
    point = np.array([geometry.TARGET_RADIUS, geometry.TARGET_CENTER_XY[1], geometry.TARGET_HEIGHT])
    left, right = 26.50, 26.52
    midpoint = (left + right) / 2
    missile = missile_position("M1", midpoint)
    for item in strategies:
        if not item.explosion_time < midpoint < item.explosion_time + geometry.SMOKE_LIFETIME:
            continue
        rate = point_time_lipschitz(point, "M1", item, left, right)
        center_distance = point_distances(
            point, missile, np.array([item.cloud_center(midpoint)])
        )[0]
        for t in np.linspace(left, right, 9):
            distance = point_distances(
                point,
                missile_position("M1", float(t)),
                np.array([item.cloud_center(float(t))]),
            )[0]
            assert abs(float(distance - center_distance)) <= rate * abs(float(t - midpoint)) + 1e-8

    # A time block crossing an activation event must never be certified from
    # the midpoint active set.
    event = min(item.explosion_time for item in strategies if item.explosion_time > 1e-3)
    status, _, _ = interval_surface_status(
        "M1", event - 1e-4, event + 1e-4, strategies, 1_000
    )
    assert status == "ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(__file__).with_name("q5_block_attacked_plan_v2.json"),
    )
    parser.add_argument("--missile", choices=["all", *q5.MISSILES], default="all")
    parser.add_argument("--space-tol", type=float, default=0.03)
    parser.add_argument("--time-tol", type=float, default=0.002)
    parser.add_argument("--max-cells", type=int, default=50_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q5_joint_certificate.json"),
    )
    parser.add_argument("--surface-time", type=float)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--outer-audit", action="store_true")
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(__file__).with_name("q5_block_attacked_validation_v2.json"),
    )
    parser.add_argument(
        "--bomb-table",
        type=Path,
        default=Path(__file__).with_name("q5_bomb_table.json"),
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    strategies = [q5.strategy_from_record(item) for item in plan["bombs"]]
    q5.validate(strategies)
    self_test(strategies)
    if args.targeted:
        record = {
            "plan": str(args.plan.resolve()),
            "validation": str(args.validation.resolve()),
            "bomb_table": str(args.bomb_table.resolve()),
            "targeted": targeted_certificate(
                args.validation, args.bomb_table, strategies, args.max_cells
            ),
        }
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return
    if args.hybrid:
        record = {
            "plan": str(args.plan.resolve()),
            "validation": str(args.validation.resolve()),
            "bomb_table": str(args.bomb_table.resolve()),
            "time_tolerance": args.time_tol,
            "hybrid": hybrid_certificate(
                args.validation,
                args.bomb_table,
                strategies,
                args.time_tol,
                args.max_cells,
            ),
        }
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return
    if args.outer_audit:
        validation = json.loads(args.validation.read_text(encoding="utf-8"))
        dense = validation["resolutions"]["dense"]["metrics"]["intervals"]
        names = list(q5.MISSILES) if args.missile == "all" else [args.missile]
        certificates = [
            outer_certificate(
                name,
                strategies,
                [tuple(pair) for pair in dense[name]],
                args.time_tol,
                args.max_cells,
            )
            for name in names
        ]
        record = {
            "plan": str(args.plan.resolve()),
            "validation": str(args.validation.resolve()),
            "time_tolerance": args.time_tol,
            "certificates": certificates,
            "total_upper": sum(float(item["duration_upper"]) for item in certificates),
        }
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return
    names = list(q5.MISSILES) if args.missile == "all" else [args.missile]
    if args.surface_time is not None:
        for name in names:
            print(name, surface_bounds(name, args.surface_time, strategies, args.space_tol, args.max_cells))
        return
    certificates = [
        certify_missile(name, strategies, args.space_tol, args.time_tol, args.max_cells)
        for name in names
    ]
    record = {
        "plan": str(args.plan.resolve()),
        "space_tolerance": args.space_tol,
        "time_tolerance": args.time_tol,
        "time_lipschitz": TIME_LIPSCHITZ,
        "certificates": certificates,
        "total_lower": sum(float(item["duration_lower"]) for item in certificates),
        "total_upper": sum(float(item["duration_upper"]) for item in certificates),
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
