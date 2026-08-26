"""CUMCM 2025 A/Q1 under the team's strict full-occlusion definition."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


MISSILE_0 = np.array([20_000.0, 0.0, 2_000.0])
MISSILE_SPEED = 300.0
UAV_0 = np.array([17_800.0, 0.0, 1_800.0])
UAV_SPEED = 120.0
RELEASE_TIME = 1.5
FUSE_DELAY = 3.6
EXPLOSION_TIME = RELEASE_TIME + FUSE_DELAY
GRAVITY = 9.8
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_LIFETIME = 20.0
TARGET_CENTER_XY = np.array([0.0, 200.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0

MISSILE_DIRECTION = -MISSILE_0 / np.linalg.norm(MISSILE_0)
MISSILE_HIT_TIME = np.linalg.norm(MISSILE_0) / MISSILE_SPEED
UAV_DIRECTION = np.array([-1.0, 0.0, 0.0])
RELEASE_POINT = UAV_0 + UAV_SPEED * RELEASE_TIME * UAV_DIRECTION
EXPLOSION_POINT = (
    RELEASE_POINT
    + UAV_SPEED * FUSE_DELAY * UAV_DIRECTION
    - 0.5 * GRAVITY * FUSE_DELAY**2 * np.array([0.0, 0.0, 1.0])
)


def missile_position(t: float) -> np.ndarray:
    return MISSILE_0 + MISSILE_SPEED * t * MISSILE_DIRECTION


def smoke_center(t: float) -> np.ndarray:
    return EXPLOSION_POINT - SMOKE_SINK_SPEED * (t - EXPLOSION_TIME) * np.array([0.0, 0.0, 1.0])


def segment_distance(m: np.ndarray, p: np.ndarray, c: np.ndarray) -> float:
    v = p - m
    lam = float(np.clip(np.dot(c - m, v) / np.dot(v, v), 0.0, 1.0))
    return float(np.linalg.norm(m + lam * v - c))


def violation(p: np.ndarray, c: np.ndarray, kappa: float) -> float:
    cp = float(np.dot(c, p))
    if cp <= kappa:
        return float(np.dot(p - c, p - c) - SMOKE_RADIUS**2)
    return float(np.dot(p, p) - cp * cp / kappa)


def linear_trig_roots(a: float, b: float, rhs: float) -> list[float]:
    amplitude = math.hypot(a, b)
    if amplitude < 1e-12 or abs(rhs) > amplitude + 1e-10:
        return []
    ratio = float(np.clip(rhs / amplitude, -1.0, 1.0))
    phase = math.atan2(b, a)
    offset = math.acos(ratio)
    return [(phase - offset) % (2 * math.pi), (phase + offset) % (2 * math.pi)]


def cone_stationary_angles(q: np.ndarray, c: np.ndarray, kappa: float) -> list[float]:
    r = TARGET_RADIUS
    s0 = float(np.dot(c, q))
    sc, ss = r * c[0], r * c[1]
    b1 = 2 * kappa * r * q[0] - 2 * s0 * sc
    b2 = 2 * kappa * r * q[1] - 2 * s0 * ss
    b3 = -(sc * sc - ss * ss) / 2
    b4 = -sc * ss
    # tan(phi/2) turns G'(phi)=0 into this quartic.
    coeff = np.array(
        [-b2 + 2 * b4, -2 * b1 + 8 * b3, -12 * b4, -2 * b1 - 8 * b3, b2 + 2 * b4],
        dtype=float,
    )
    scale = max(1.0, float(np.max(np.abs(coeff))))
    coeff = np.trim_zeros(np.where(np.abs(coeff) < 1e-13 * scale, 0.0, coeff), "f")
    if len(coeff) <= 1:
        return []
    angles: list[float] = []
    for root in np.roots(coeff):
        if abs(root.imag) <= 1e-8 * max(1.0, abs(root.real)):
            angles.append((2 * math.atan(float(root.real))) % (2 * math.pi))
    return angles


def ring_max_violation(m: np.ndarray, cloud: np.ndarray, z: float) -> tuple[float, np.ndarray]:
    c = cloud - m
    d2 = float(np.dot(c, c))
    if d2 <= SMOKE_RADIUS**2:
        return -math.inf, np.array([TARGET_RADIUS, 200.0, z])
    kappa = d2 - SMOKE_RADIUS**2
    q = np.array([0.0, 200.0, z]) - m
    r = TARGET_RADIUS
    s0 = float(np.dot(c, q))
    sc, ss = r * c[0], r * c[1]

    candidates = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    boundaries = linear_trig_roots(sc, ss, kappa - s0)
    candidates.extend(boundaries)

    # Stationary points of the spherical-cap branch.
    a1, a2 = 2 * r * (q[0] - c[0]), 2 * r * (q[1] - c[1])
    if math.hypot(a1, a2) > 1e-12:
        phi = math.atan2(a2, a1) % (2 * math.pi)
        candidates.extend([phi, (phi + math.pi) % (2 * math.pi)])

    # Stationary points of the tangent-cone branch.
    candidates.extend(cone_stationary_angles(q, c, kappa))

    # One representative per branch interval handles degenerate derivatives.
    cuts = sorted({round(x % (2 * math.pi), 14) for x in boundaries})
    if cuts:
        wrapped = cuts + [cuts[0] + 2 * math.pi]
        candidates.extend([((a + b) / 2) % (2 * math.pi) for a, b in zip(wrapped, wrapped[1:])])

    best_v = -math.inf
    best_p = np.zeros(3)
    for phi in candidates:
        p_abs = np.array([r * math.cos(phi), 200.0 + r * math.sin(phi), z])
        value = violation(p_abs - m, c, kappa)
        if value > best_v:
            best_v, best_p = value, p_abs
    return best_v, best_p


def c_observer(t: float) -> tuple[float, np.ndarray, float]:
    """Return exact piecewise violation maximum, witness point and direct distance."""
    m, cloud = missile_position(t), smoke_center(t)
    results = [ring_max_violation(m, cloud, z) for z in (0.0, TARGET_HEIGHT)]
    vmax, witness = max(results, key=lambda item: item[0])
    return vmax, witness, segment_distance(m, witness, cloud)


def b_bounds(t: float, n: int = 4096) -> tuple[float, float]:
    """Certified lower/upper bounds for the worst sight-line distance."""
    phi = 2 * math.pi * np.arange(n) / n
    x = TARGET_RADIUS * np.cos(phi)
    y = 200.0 + TARGET_RADIUS * np.sin(phi)
    lower = 0.0
    m, cloud = missile_position(t), smoke_center(t)
    for z in (0.0, TARGET_HEIGHT):
        points = np.column_stack((x, y, np.full(n, z)))
        v = points - m
        lam = np.clip(((cloud - m) @ v.T) / np.einsum("ij,ij->i", v, v), 0.0, 1.0)
        closest = m + lam[:, None] * v
        lower = max(lower, float(np.max(np.linalg.norm(closest - cloud, axis=1))))
    angular_gap_bound = 2 * TARGET_RADIUS * math.sin(math.pi / (2 * n))
    return lower, lower + angular_gap_bound


@dataclass
class TimeCertificate:
    covered: list[tuple[float, float]]
    unresolved: list[tuple[float, float]]

    @property
    def lower_duration(self) -> float:
        return sum(b - a for a, b in self.covered)

    @property
    def upper_duration(self) -> float:
        return self.lower_duration + sum(b - a for a, b in self.unresolved)


def merge(intervals: list[tuple[float, float]], tol: float = 1e-12) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [list(pair) for pair in sorted(intervals)]
    out = [merged[0]]
    for a, b in merged[1:]:
        if a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(float(a), float(b)) for a, b in out]


def certify_time(eps_t: float = 2e-4, n_ring: int = 2048) -> TimeCertificate:
    start = EXPLOSION_TIME
    end = min(EXPLOSION_TIME + SMOKE_LIFETIME, MISSILE_HIT_TIME)
    stack = [(start, end)]
    covered: list[tuple[float, float]] = []
    unresolved: list[tuple[float, float]] = []
    temporal_lipschitz = MISSILE_SPEED + SMOKE_SINK_SPEED
    while stack:
        a, b = stack.pop()
        mid, half = (a + b) / 2, (b - a) / 2
        lower, upper = b_bounds(mid, n_ring)
        if upper + temporal_lipschitz * half <= SMOKE_RADIUS:
            covered.append((a, b))
        elif lower - temporal_lipschitz * half > SMOKE_RADIUS:
            continue
        elif b - a <= eps_t:
            unresolved.append((a, b))
        else:
            stack.extend([(a, mid), (mid, b)])
    return TimeCertificate(merge(covered), merge(unresolved))


def bisect_boundary(a: float, b: float, iterations: int = 70) -> float:
    fa = c_observer(a)[0]
    fb = c_observer(b)[0]
    if fa * fb > 0:
        raise ValueError("boundary is not bracketed")
    for _ in range(iterations):
        mid = (a + b) / 2
        fm = c_observer(mid)[0]
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return (a + b) / 2


def c_intervals(step: float = 0.01) -> list[tuple[float, float]]:
    start = EXPLOSION_TIME
    end = min(EXPLOSION_TIME + SMOKE_LIFETIME, MISSILE_HIT_TIME)
    times = np.arange(start, end + step / 2, step)
    values = np.array([c_observer(float(t))[0] for t in times])
    roots: list[float] = []
    for idx in range(len(times) - 1):
        if values[idx] * values[idx + 1] < 0:
            roots.append(bisect_boundary(float(times[idx]), float(times[idx + 1])))
    cuts = [start] + roots + [end]
    intervals = []
    for a, b in zip(cuts, cuts[1:]):
        if c_observer((a + b) / 2)[0] <= 0:
            intervals.append((a, b))
    return intervals


def main() -> None:
    analytic = c_intervals()
    certificate = certify_time()

    contradictions = 0
    for t in np.linspace(EXPLOSION_TIME, EXPLOSION_TIME + SMOKE_LIFETIME, 201):
        c_pass = c_observer(float(t))[0] <= 0
        lower, upper = b_bounds(float(t), 8192)
        if (upper <= SMOKE_RADIUS and not c_pass) or (lower > SMOKE_RADIUS and c_pass):
            contradictions += 1

    print(f"release_point={RELEASE_POINT.tolist()}")
    print(f"explosion_time={EXPLOSION_TIME:.9f}")
    print(f"explosion_point={EXPLOSION_POINT.tolist()}")
    print(f"missile_hit_time={MISSILE_HIT_TIME:.9f}")
    print(f"c_intervals={analytic}")
    print(f"c_duration={sum(b - a for a, b in analytic):.9f}")
    print(f"certified_covered={certificate.covered}")
    print(f"certified_unresolved={certificate.unresolved}")
    print(f"certified_duration=[{certificate.lower_duration:.9f}, {certificate.upper_duration:.9f}]")
    print(f"C_vs_B_contradictions={contradictions}")

    assert contradictions == 0
    assert certificate.lower_duration <= sum(b - a for a, b in analytic) <= certificate.upper_duration


if __name__ == "__main__":
    main()
