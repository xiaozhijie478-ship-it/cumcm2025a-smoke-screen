"""Reproducible upper certificate for Q4's independent three-window branch."""

from __future__ import annotations

import json
from pathlib import Path

import q2_global_search as q2_global
import q4_optimize as q4


THRESHOLDS = {"FY1": 4.589, "FY2": 4.000, "FY3": 3.250}
OUTPUT = Path(__file__).with_name("q4_independent_upper_certificate.json")


def main() -> None:
    strategies = {item.name: item for item in q4.final_total_candidate()}
    certificates: dict[str, object] = {}
    current_total = 0.0
    upper_total = 0.0
    for name, threshold in THRESHOLDS.items():
        intervals = q4.q2.c_intervals(strategies[name], 0.002)
        current = sum(end - start for start, end in intervals)
        current_total += current
        upper_total += threshold
        q2_global.UAV_0 = q4.UAVS[name]
        result = q2_global.certify_relaxed_upper(threshold)
        if result["remaining"] or result["unresolved"]:
            raise RuntimeError(f"{name} threshold was not certified: {result}")
        certificates[name] = {
            "current_intervals": intervals,
            "current_duration": current,
            "strict_upper": threshold,
            "gap": threshold - current,
            "conic_certificate": result,
        }
    payload = {
        "scope": "three independent single-ball full-occlusion windows",
        "current_total": current_total,
        "strict_upper_total": upper_total,
        "absolute_gap": upper_total - current_total,
        "relative_gap_to_upper": (upper_total - current_total) / upper_total,
        "certificates": certificates,
        "limitations": [
            "The certificate does not cover time instants that require two balls jointly.",
            "Interpreting each single-ball total as one interval uses the audited but not analytically proved interval-connectedness property.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
