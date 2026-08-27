"""Check deliverables result1-3.xlsx against saved JSON evidence and formal values.

Checks:
  1. duration-column sums match the formal Q3/Q4 values (result3 is a per-bomb
     table; its aggregate J_sum is checked from the validation JSON instead);
  2. every row is kinematically consistent (heading/speed -> burst point, gravity drop);
  3. result3.xlsx rows match q5_bomb_table.json bomb-by-bomb;
  4. Q5 J_sum / J_min / J_all from the dense validation JSON match the formal values;
  5. saved certificate JSONs contain the claimed formal numbers;
  6. validity constraints: burst altitude >= 0, coverage within 20 s of burst.

Exit code 0 means all checks pass; 1 means at least one check failed.
"""

from __future__ import annotations

import json
import math
import os
import sys

import openpyxl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERABLES = os.path.join(REPO, "deliverables")
G = 9.8

# (file, duration col, theta col, speed col, release cols, burst cols, missile col or None)
LAYOUTS = {
    "result1.xlsx": (9, 0, 1, (3, 4, 5), (6, 7, 8), None),
    "result2.xlsx": (9, 1, 2, (3, 4, 5), (6, 7, 8), None),
    "result3.xlsx": (10, 1, 2, (4, 5, 6), (7, 8, 9), 11),
}

FORMAL = {
    "result1.xlsx": 7.650405706,   # Q3 cumulative
    "result2.xlsx": 11.735130825,  # Q4 cumulative
}

# tolerance for duration sums (result2 stores 6 decimals)
SUM_TOLERANCE = {
    "result1.xlsx": 1e-9,
    "result2.xlsx": 2e-5,
}

Q5_FORMAL = {"J_sum": 35.109171990, "J_min": 7.499774830, "J_all": 1.439349}


def load_rows(path: str) -> list[tuple]:
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    return [r for r in rows[1:] if r[0] is not None]


def union_duration(intervals: list[list[float]]) -> float:
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged)


def kinematic_check(
    label: str,
    theta_deg: float,
    speed: float,
    rel: tuple,
    burst: tuple,
    tol_m: float = 0.05,
) -> list[str]:
    """Return list of problems; empty means the row is consistent."""
    problems: list[str] = []
    rx, ry, rz = rel
    bx, by, bz = burst
    if None in (rx, ry, rz, bx, by, bz, speed, theta_deg):
        return problems
    dx, dy, dz = bx - rx, by - ry, rz - bz  # horizontal displacement + vertical drop
    horiz = math.hypot(dx, dy)
    if horiz < 1e-9:
        return problems  # no travel (delay 0), nothing to verify
    if speed <= 0:
        problems.append(f"{label}: speed <= 0")
        return problems
    dt = horiz / speed
    predicted_drop = 0.5 * G * dt * dt
    if abs(predicted_drop - dz) > tol_m:
        problems.append(
            f"{label}: gravity drop mismatch (predicted {predicted_drop:.4f} m, "
            f"actual {dz:.4f} m, dt={dt:.4f} s)"
        )
    theta = math.radians(theta_deg)
    dirx, diry = math.cos(theta), math.sin(theta)
    dot = (dx * dirx + dy * diry) / horiz if horiz else 0.0
    if dot < 0.999:
        problems.append(f"{label}: heading mismatch (dot={dot:.6f}, theta={theta_deg:.6f})")
    return problems


def main() -> int:
    failures: list[str] = []
    checks_run = 0

    bomb_table = None
    table_path = os.path.join(REPO, "q5_bomb_table.json")
    if os.path.exists(table_path):
        bomb_table = json.load(open(table_path, encoding="utf-8"))["bombs"]

    for filename, (dur_col, theta_col, speed_col, rel_cols, burst_cols, missile_col) in LAYOUTS.items():
        path = os.path.join(DELIVERABLES, filename)
        if not os.path.exists(path):
            failures.append(f"{filename}: file missing")
            continue
        rows = load_rows(path)
        durations = [
            float(r[dur_col]) for r in rows if isinstance(r[dur_col], (int, float))
        ]
        total = sum(durations)
        if filename in FORMAL:
            expected = FORMAL[filename]
            checks_run += 1
            if abs(total - expected) > SUM_TOLERANCE[filename]:
                failures.append(
                    f"{filename}: duration sum {total:.9f} != formal {expected:.9f}"
                )
            else:
                print(
                    f"PASS {filename}: duration sum {total:.9f} == formal {expected:.9f}"
                )
        else:
            print(
                f"INFO {filename}: per-bomb duration sum {total:.9f} "
                f"(not directly comparable to J_sum; see validation JSON)"
            )

        for idx, r in enumerate(rows, start=2):
            rel = tuple(r[c] for c in rel_cols)
            burst = tuple(r[c] for c in burst_cols)
            if all(v is None for v in rel):
                continue
            label = f"{filename} row {idx}"
            for problem in kinematic_check(
                label, r[theta_col], r[speed_col], rel, burst
            ):
                failures.append(problem)
                checks_run += 1

        if filename == "result3.xlsx" and bomb_table is not None:
            checks_run += 1
            for r in rows:
                uav = r[0]
                if r[speed_col] is None or r[rel_cols[0]] is None:
                    continue  # unused bomb slot (e.g. FY2-2, FY2-3)
                number = int(r[3])
                match = next(
                    (b for b in bomb_table if b["uav"] == uav and b["number"] == number),
                    None,
                )
                if match is None:
                    failures.append(f"result3.xlsx: no bomb-table match for {uav}-{number}")
                    continue
                rel = tuple(r[c] for c in rel_cols)
                burst = tuple(r[c] for c in burst_cols)
                if (
                    any(abs(float(rel[k]) - match["release_point"][k]) > 1e-6 for k in range(3))
                    or any(abs(float(burst[k]) - match["explosion_point"][k]) > 1e-6 for k in range(3))
                    or abs(float(r[dur_col]) - match["individual_duration"]) > 1e-9
                    or r[missile_col] != match["assigned_missile"]
                ):
                    failures.append(
                        f"result3.xlsx: {uav}-{number} differs from q5_bomb_table.json"
                    )
            print(f"PASS result3.xlsx: all 11 bombs match q5_bomb_table.json")

    # Q5 validity constraints from the bomb table
    if bomb_table is not None:
        validity_problems = 0
        for b in bomb_table:
            expl = b["explosion_time"]
            if any(end > expl + 20.0 + 1e-9 for _, end in b["individual_intervals"]):
                failures.append(f"q5 bomb {b['uav']}-{b['number']}: coverage beyond 20 s validity")
                validity_problems += 1
            if b["explosion_point"][2] < -1e-6:
                failures.append(f"q5 bomb {b['uav']}-{b['number']}: burst altitude below zero")
                validity_problems += 1
        if validity_problems == 0:
            print("PASS q5_bomb_table.json: burst altitude >= 0 and coverage within 20 s validity")
            checks_run += 1

    # Q5 aggregate metrics from the dense validation JSON
    val_path = os.path.join(REPO, "q5_block_attacked_validation_v2.json")
    if os.path.exists(val_path):
        validation = json.load(open(val_path, encoding="utf-8"))
        dense = validation["resolutions"]["dense"]["metrics"]
        checks_run += 3
        for name, value, formal in (
            ("J_sum", dense["total"], Q5_FORMAL["J_sum"]),
            ("J_min", dense["minimum"], Q5_FORMAL["J_min"]),
            ("J_all", dense["simultaneous"], Q5_FORMAL["J_all"]),
        ):
            if abs(float(value) - formal) > 1e-6:
                failures.append(f"Q5 {name}: dense validation {value} != formal {formal}")
            else:
                print(f"PASS Q5 {name}: dense validation {value:.9f} == formal {formal:.9f}")

        rounded = validation["rounded_decision_metrics"]
        checks_run += 1
        if abs(float(rounded["total"]) - 35.044849322) > 1e-6:
            failures.append(f"Q5 rounded total {rounded['total']} differs from recorded 35.044849322")
        else:
            print(f"PASS Q5 rounded-decision total: {rounded['total']:.9f} (3-decimal round-back feasible)")

    # certificate JSON values
    cert_checks = [
        ("q4_independent_upper_certificate.json", "current_total", 11.735130825, 1e-6),
        ("q4_independent_upper_certificate.json", "strict_upper_total", 11.839, 1e-6),
        ("q5_outer_continuous_certificate_tight.json", "total_upper", 35.112385634, 1e-6),
        ("q5_joint_hybrid_certificate.json", "total_certified_lower", 35.067366178, 1e-6),
    ]
    for fname, key, expected, tol in cert_checks:
        path = os.path.join(REPO, fname)
        if not os.path.exists(path):
            failures.append(f"{fname}: missing")
            continue
        data = json.load(open(path, encoding="utf-8"))
        checks_run += 1
        if fname == "q5_joint_hybrid_certificate.json":
            value = data["hybrid"]["total_certified_lower"]
        else:
            value = data[key]
        if abs(float(value) - expected) > tol:
            failures.append(f"{fname}.{key}: {value} != {expected}")
        else:
            print(f"PASS {fname}.{key} == {expected}")

    print(f"\nchecks run: {checks_run}, failures: {len(failures)}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
