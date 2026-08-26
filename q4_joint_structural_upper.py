"""Structural upper bound for the complete Q4 joint-coverage domain.

FY2 is the articulation cloud: FY1 and FY3 can never contribute at the same
time.  Outside FY2's possible-contribution set, any complete occlusion must be
provided by FY1 or FY3 alone.  We therefore bound

    J4 <= U(FY1 single) + span(FY2 can touch any LOS) + U(FY3 single).

The two single-ball bounds use the audited connectedness assumption documented
in ``q4_independent_upper_certificate.json``.  The FY2 span bound itself does
not assume connectedness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import q1_strict_occlusion as geometry
import q4_optimize as q4


STEP = 1e-4
FY2 = q4.UAVS["FY2"][:2]
TARGET = np.array(
    [geometry.TARGET_CENTER_XY[0], geometry.TARGET_CENTER_XY[1]], dtype=float
)
TARGET_RADIUS = geometry.TARGET_RADIUS
SMOKE_RADIUS = geometry.SMOKE_RADIUS
VMAX = q4.q2.SPEED_BOUNDS[1]
MISSILE_HORIZONTAL_SPEED = abs(
    geometry.MISSILE_SPEED * geometry.MISSILE_DIRECTION[0]
)
SINGLE_UPPERS = {"FY1": 4.589, "FY3": 3.250}
OUTPUT = Path(__file__).with_name("q4_joint_structural_upper.json")


def point_segment_distances(
    point: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    directions = ends - starts
    fractions = np.clip(
        np.einsum("ni,ni->n", point - starts, directions)
        / np.einsum("ni,ni->n", directions, directions),
        0.0,
        1.0,
    )
    closest = starts + fractions[:, None] * directions
    return np.linalg.norm(point - closest, axis=1)


def horizontal_hull_distances(times: np.ndarray) -> np.ndarray:
    """Exact FY2 distance to conv(horizontal missile point, target disk)."""
    missiles = (
        geometry.MISSILE_0[:2]
        + geometry.MISSILE_SPEED
        * times[:, None]
        * geometry.MISSILE_DIRECTION[:2]
    )
    vectors = missiles - TARGET
    squared = np.einsum("ni,ni->n", vectors, vectors)
    if np.any(squared <= TARGET_RADIUS**2):
        raise ValueError("missile horizontal point entered the target disk")
    perpendicular = np.column_stack((-vectors[:, 1], vectors[:, 0]))
    radial = (TARGET_RADIUS**2 / squared)[:, None] * vectors
    tangent = (
        TARGET_RADIUS * np.sqrt(squared - TARGET_RADIUS**2) / squared
    )[:, None] * perpendicular
    first = TARGET + radial + tangent
    second = TARGET + radial - tangent

    disk_distance = max(0.0, float(np.linalg.norm(FY2 - TARGET) - TARGET_RADIUS))
    boundary_distance = np.minimum(
        point_segment_distances(FY2, missiles, first),
        point_segment_distances(FY2, missiles, second),
    )
    distances = np.minimum(boundary_distance, disk_distance)
    # FY2 is outside every horizontal convex hull in this problem.  The check
    # protects the boundary-distance formula from being reused silently where
    # an interior point should have distance zero.
    if float(np.min(distances)) <= 0.0:
        raise ValueError("FY2 lies inside a horizontal sight hull")
    return distances


def certificate() -> dict[str, float | int | dict[str, float]]:
    times = np.arange(0.0, geometry.MISSILE_HIT_TIME + STEP / 2, STEP)
    times[-1] = geometry.MISSILE_HIT_TIME
    hull_distance = horizontal_hull_distances(times)
    required_travel = np.maximum(0.0, hull_distance - SMOKE_RADIUS)
    raw_spans = np.minimum(
        geometry.SMOKE_LIFETIME,
        np.maximum(0.0, times - required_travel / VMAX),
    )
    index = int(np.argmax(raw_spans))

    # Distance to a convex set moving at horizontal speed <= |vx| is
    # |vx|-Lipschitz.  Hence b-d(b)/vmax is Lipschitz with this constant; the
    # half-grid guard turns the sampled maximum into a continuous-time upper.
    lipschitz = 1.0 + MISSILE_HORIZONTAL_SPEED / VMAX
    grid_guard = lipschitz * STEP / 2
    floating_guard = 1e-9
    fy2_span_upper = min(
        geometry.SMOKE_LIFETIME,
        float(raw_spans[index] + grid_guard + floating_guard),
    )
    total_upper = SINGLE_UPPERS["FY1"] + fy2_span_upper + SINGLE_UPPERS["FY3"]
    return {
        "grid_step": STEP,
        "grid_points": len(times),
        "lipschitz_constant": lipschitz,
        "grid_guard": grid_guard,
        "sample_argmax_time": float(times[index]),
        "sample_max_span": float(raw_spans[index]),
        "fy2_possible_contribution_span_upper": fy2_span_upper,
        "single_duration_uppers": SINGLE_UPPERS,
        "q4_complete_joint_upper_under_single_connectedness": total_upper,
    }


def self_test() -> None:
    starts = np.array([[0.0, 0.0], [0.0, 0.0]])
    ends = np.array([[2.0, 0.0], [0.0, 2.0]])
    values = point_segment_distances(np.array([1.0, 1.0]), starts, ends)
    assert np.allclose(values, [1.0, 1.0])
    result = certificate()
    assert 0.0 < result["fy2_possible_contribution_span_upper"] <= 20.0
    assert result["q4_complete_joint_upper_under_single_connectedness"] < 27.839


def main() -> None:
    self_test()
    result = certificate()
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
