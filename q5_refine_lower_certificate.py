"""Refine only the unresolved cells of a Q5 continuous lower certificate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import q5_certify_joint as cert
import q5_optimize as q5


EPS = 1e-9


def intervals(values: object) -> list[tuple[float, float]]:
    return [(float(left), float(right)) for left, right in values]  # type: ignore[misc]


def duration(values: list[tuple[float, float]]) -> float:
    return sum(right - left for left, right in values)


def overlap_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def assert_partition(
    original: list[tuple[float, float]], refined: dict[str, object], label: str
) -> None:
    parts = [
        cert.merge(intervals(refined[name]))
        for name in ("covered", "unresolved", "rejected")
    ]
    for left, right in itertools.combinations(parts, 2):
        if overlap_duration(left, right) > EPS:
            raise ValueError(f"{label}: refined classes overlap")
    expected = cert.merge(original)
    actual = cert.merge([item for part in parts for item in part])
    if len(expected) != len(actual) or any(
        abs(a - c) > EPS or abs(b - d) > EPS
        for (a, b), (c, d) in zip(expected, actual)
    ):
        raise ValueError(f"{label}: refined classes do not partition old unresolved cells")


def refine_certificate(
    old: dict[str, object],
    missile: str,
    strategies: list[q5.BombStrategy],
    time_tolerance: float,
    max_cells: int,
    label: str,
) -> tuple[dict[str, object], dict[str, float]]:
    old_unresolved = intervals(old["unresolved"])
    new = cert.certify_candidate_intervals(
        missile, old_unresolved, strategies, time_tolerance, max_cells
    )
    assert_partition(old_unresolved, new, label)

    covered = cert.merge(intervals(old["covered"]) + intervals(new["covered"]))
    unresolved = cert.merge(intervals(new["unresolved"]))
    rejected = cert.merge(intervals(old["rejected"]) + intervals(new["rejected"]))
    result = copy.deepcopy(old)
    result.update(
        {
            "covered": covered,
            "unresolved": unresolved,
            "rejected": rejected,
            "duration_lower": duration(covered),
            "duration_unresolved": duration(unresolved),
            "nodes": int(old.get("nodes", 0)) + int(new["nodes"]),
            "peak_surface_cells": max(
                int(old.get("peak_surface_cells", 0)), int(new["peak_surface_cells"])
            ),
        }
    )
    if float(result["duration_lower"]) + EPS < float(old["duration_lower"]):
        raise ValueError(f"{label}: certified lower bound decreased")
    return result, {
        "source_unresolved": duration(old_unresolved),
        "remaining_unresolved": duration(unresolved),
        "nodes": float(new["nodes"]),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument(
        "--plan", type=Path, default=Path(__file__).with_name("q5_final_plan.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-tol", type=float, default=0.00025)
    parser.add_argument("--max-cells", type=int, default=30_000)
    args = parser.parse_args()
    if args.time_tol <= 0 or args.max_cells <= 0:
        parser.error("--time-tol and --max-cells must be positive")
    output = args.output or args.certificate.with_name(
        f"{args.certificate.stem}_refined.json"
    )
    if output.resolve() == args.certificate.resolve():
        parser.error("refusing to overwrite the source certificate")

    source_bytes = args.certificate.read_bytes()
    source = json.loads(source_bytes)
    original_input_sha256 = copy.deepcopy(source.get("input_sha256"))
    plan_hash = sha256(args.plan)
    expected_plan_hash = (source.get("input_sha256") or {}).get("plan")
    if expected_plan_hash and plan_hash != expected_plan_hash:
        raise ValueError("plan SHA-256 does not match the source certificate")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    strategies = [q5.strategy_from_record(item) for item in plan["bombs"]]
    q5.validate(strategies)
    by_key = {f"{item.uav}-{item.number}": item for item in strategies}
    if len(by_key) != len(strategies):
        raise ValueError("plan contains duplicate UAV/bomb keys")

    record = copy.deepcopy(source)
    source_total = float(record["hybrid"]["total_certified_lower"])
    audit = {"source_unresolved": 0.0, "remaining_unresolved": 0.0, "nodes": 0.0}
    for missile, missile_record in record["hybrid"]["by_missile"].items():
        if missile not in q5.MISSILES:
            raise ValueError(f"unknown missile {missile}")
        single = missile_record["single_certificate"]
        single_covered: list[tuple[float, float]] = []
        for key, bomb_record in single["by_bomb"].items():
            strategy = by_key.get(key)
            if strategy is None or strategy.assigned_missile != missile:
                raise ValueError(f"{missile}/{key}: plan assignment mismatch")
            refined, stats = refine_certificate(
                bomb_record["certificate"],
                missile,
                [strategy],
                args.time_tol,
                args.max_cells,
                f"{missile}/{key}/single",
            )
            bomb_record["certificate"] = refined
            single_covered.extend(intervals(refined["covered"]))
            for name in audit:
                audit[name] += stats[name]

        single_union = cert.merge(single_covered)
        single["covered_union"] = single_union
        single["duration_lower"] = duration(single_union)
        missile_record["certified_single_union"] = single_union
        missile_record["single_duration"] = duration(single_union)

        joint, stats = refine_certificate(
            missile_record["joint_only_certificate"],
            missile,
            strategies,
            args.time_tol,
            args.max_cells,
            f"{missile}/joint",
        )
        missile_record["joint_only_certificate"] = joint
        for name in audit:
            audit[name] += stats[name]
        certified = cert.merge(single_union + intervals(joint["covered"]))
        missile_record["certified_intervals"] = certified
        missile_record["certified_duration_lower"] = duration(certified)

    record["hybrid"]["total_certified_lower"] = sum(
        float(item["certified_duration_lower"])
        for item in record["hybrid"]["by_missile"].values()
    )
    record["hybrid"]["total_dense"] = sum(
        float(item["dense_duration"])
        for item in record["hybrid"]["by_missile"].values()
    )
    refined_total = float(record["hybrid"]["total_certified_lower"])
    if refined_total + EPS < source_total:
        raise ValueError("total certified lower bound decreased")
    if record.get("input_sha256") != original_input_sha256:
        raise ValueError("input_sha256 changed during refinement")

    record["refinement"] = {
        "method": "re-certify source unresolved cells only",
        "source_certificate": str(args.certificate.resolve()),
        "source_certificate_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_hash,
        "time_tolerance": args.time_tol,
        "max_cells": args.max_cells,
        "source_total_certified_lower": source_total,
        "refined_total_certified_lower": refined_total,
        "lower_bound_gain": refined_total - source_total,
        "source_unresolved_duration": audit["source_unresolved"],
        "remaining_unresolved_duration": audit["remaining_unresolved"],
        "refinement_nodes": int(audit["nodes"]),
    }
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **record["refinement"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
