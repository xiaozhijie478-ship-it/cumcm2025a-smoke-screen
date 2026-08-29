"""Derive J_sum, J_min and J_all bounds for one fixed Q5 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import q5_optimize as q5


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    result: list[list[float]] = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if result and left <= result[-1][1] + 1e-12:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return [(left, right) for left, right in result]


def intersect(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        left = max(first[i][0], second[j][0])
        right = min(first[i][1], second[j][1])
        if right > left:
            result.append((left, right))
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return merge(result)


def subtract(
    whole: list[tuple[float, float]], remove: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    result = []
    for left, right in whole:
        cursor = left
        for a, b in remove:
            if b <= cursor or a >= right:
                continue
            if a > cursor:
                result.append((cursor, min(a, right)))
            cursor = max(cursor, b)
            if cursor >= right:
                break
        if cursor < right:
            result.append((cursor, right))
    return merge(result)


def measure(intervals: list[tuple[float, float]]) -> float:
    return sum(right - left for left, right in intervals)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_partition(
    candidates: list[tuple[float, float]], certificate: dict[str, object], label: str
) -> dict[str, float]:
    classes = {
        name: merge([tuple(pair) for pair in certificate[name]])
        for name in ("covered", "unresolved", "rejected")
    }
    candidate_union = merge(candidates)
    classified_union = merge([pair for intervals in classes.values() for pair in intervals])
    missing = measure(subtract(candidate_union, classified_union))
    outside = measure(subtract(classified_union, candidate_union))
    overlap = sum(measure(intervals) for intervals in classes.values()) - measure(
        classified_union
    )
    if max(missing, outside, overlap) > 1e-8:
        raise RuntimeError(
            f"{label} is not a disjoint complete partition: "
            f"missing={missing}, outside={outside}, overlap={overlap}"
        )
    return {"missing": missing, "outside": outside, "overlap": overlap}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower",
        type=Path,
        default=Path(__file__).with_name("q5_final_continuous_lower.json"),
    )
    parser.add_argument(
        "--upper",
        type=Path,
        default=Path(__file__).with_name("q5_final_continuous_upper.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q5_final_aggregate_certificate.json"),
    )
    args = parser.parse_args()

    lower_record = json.loads(args.lower.read_text(encoding="utf-8"))
    upper_record = json.loads(args.upper.read_text(encoding="utf-8"))
    lower_plan = Path(lower_record["plan"]).resolve()
    upper_plan = Path(upper_record["plan"]).resolve()
    if lower_plan != upper_plan:
        raise RuntimeError(f"lower and upper certificates use different plans: {lower_plan}, {upper_plan}")
    if digest(lower_plan) != lower_record["input_sha256"]["plan"]:
        raise RuntimeError("the fixed plan no longer matches the lower certificate hash")
    if digest(upper_plan) != upper_record["input_sha256"]["plan"]:
        raise RuntimeError("the fixed plan no longer matches the upper certificate hash")
    for field in ("validation", "bomb_table"):
        source = Path(lower_record[field]).resolve()
        if digest(source) != lower_record["input_sha256"][field]:
            raise RuntimeError(f"the lower-certificate {field} input has changed")
    upper_validation = Path(upper_record["validation"]).resolve()
    if digest(upper_validation) != upper_record["input_sha256"]["validation"]:
        raise RuntimeError("the upper-certificate validation input has changed")
    if upper_validation != Path(lower_record["validation"]).resolve():
        raise RuntimeError("lower and upper certificates use different validations")
    lower_source = lower_record["hybrid"]["by_missile"]
    missile_names = list(q5.MISSILES)
    if set(lower_source) != set(missile_names):
        raise RuntimeError("lower certificate does not contain exactly M1, M2 and M3")
    upper_names = [item["missile"] for item in upper_record["certificates"]]
    if len(upper_names) != len(set(upper_names)) or set(upper_names) != set(missile_names):
        raise RuntimeError("upper certificate does not contain exactly one record per missile")
    upper_source = {item["missile"]: item for item in upper_record["certificates"]}

    lower_sets: dict[str, list[tuple[float, float]]] = {}
    upper_sets: dict[str, list[tuple[float, float]]] = {}
    by_missile: dict[str, object] = {}
    partition_checks: dict[str, object] = {}
    for name in missile_names:
        lower_item = lower_source[name]
        single_checks = {}
        single_covered = []
        for bomb, item in lower_item["single_certificate"]["by_bomb"].items():
            certificate = item["certificate"]
            single_checks[bomb] = validate_partition(
                [tuple(pair) for pair in item["padded_candidate_intervals"]],
                certificate,
                f"{name}/{bomb}",
            )
            single_covered.extend(tuple(pair) for pair in certificate["covered"])
        joint_certificate = lower_item["joint_only_certificate"]
        joint_check = validate_partition(
            [tuple(pair) for pair in lower_item["dense_joint_only_candidates"]],
            joint_certificate,
            f"{name}/joint",
        )
        reconstructed_lower = merge(
            single_covered + [tuple(pair) for pair in joint_certificate["covered"]]
        )
        lower_sets[name] = merge(
            [tuple(pair) for pair in lower_item["certified_intervals"]]
        )
        reconstruction_error = measure(subtract(lower_sets[name], reconstructed_lower)) + measure(
            subtract(reconstructed_lower, lower_sets[name])
        )
        if reconstruction_error > 1e-8:
            raise RuntimeError(f"{name} certified union cannot be reconstructed")
        partition_checks[name] = {
            "single": single_checks,
            "joint": joint_check,
            "certified_union_reconstruction_error": reconstruction_error,
        }
        upper_item = upper_source[name]
        hinted = merge([tuple(pair) for pair in upper_item["hinted_possible"]])
        whole_time = [(0.0, q5.missile_hit_time(name))]
        hinted_outside_time = measure(subtract(hinted, whole_time))
        if hinted_outside_time > 1e-8:
            raise RuntimeError(f"{name} hinted upper intervals leave the physical time domain")
        outside_candidates = subtract(whole_time, hinted)
        outside_certificate = {
            "covered": upper_item["outside_covered"],
            "unresolved": upper_item["outside_unresolved"],
            "rejected": upper_item["outside_rejected"],
        }
        upper_partition_check = validate_partition(
            outside_candidates, outside_certificate, f"{name}/upper-outside"
        )
        reconstructed_upper = merge(
            hinted
            + [tuple(pair) for pair in upper_item["outside_covered"]]
            + [tuple(pair) for pair in upper_item["outside_unresolved"]]
        )
        upper_sets[name] = merge(
            [tuple(pair) for pair in upper_item["possible_intervals"]]
        )
        upper_reconstruction_error = measure(subtract(upper_sets[name], reconstructed_upper)) + measure(
            subtract(reconstructed_upper, upper_sets[name])
        )
        if upper_reconstruction_error > 1e-8:
            raise RuntimeError(f"{name} possible upper set cannot be reconstructed")
        if abs(measure(upper_sets[name]) - float(upper_item["duration_upper"])) > 1e-8:
            raise RuntimeError(f"{name} upper duration does not match its interval union")
        partition_checks[name]["upper_outside"] = upper_partition_check
        partition_checks[name]["upper_union_reconstruction_error"] = upper_reconstruction_error
        partition_checks[name]["upper_hint_outside_time_domain"] = hinted_outside_time
        outside = subtract(lower_sets[name], upper_sets[name])
        if measure(outside) > 1e-9:
            raise RuntimeError(f"{name} lower set is not enclosed by its upper set: {outside}")
        by_missile[name] = {
            "lower_intervals": lower_sets[name],
            "upper_intervals": upper_sets[name],
            "duration_lower": measure(lower_sets[name]),
            "duration_upper": measure(upper_sets[name]),
        }

    all_lower = lower_sets[missile_names[0]]
    all_upper = upper_sets[missile_names[0]]
    for name in missile_names[1:]:
        all_lower = intersect(all_lower, lower_sets[name])
        all_upper = intersect(all_upper, upper_sets[name])

    sum_lower = sum(float(item["duration_lower"]) for item in by_missile.values())
    sum_upper = sum(float(item["duration_upper"]) for item in by_missile.values())
    if abs(sum_lower - float(lower_record["hybrid"]["total_certified_lower"])) > 1e-8:
        raise RuntimeError("lower total does not match the per-missile certified unions")
    if abs(sum_upper - float(upper_record["total_upper"])) > 1e-8:
        raise RuntimeError("upper total does not match the per-missile possible unions")
    min_lower = min(float(item["duration_lower"]) for item in by_missile.values())
    min_upper = min(float(item["duration_upper"]) for item in by_missile.values())
    payload = {
        "scope": "continuous-domain numerical certificate for one fixed 13-bomb plan; not a global optimization bound",
        "lower_certificate": str(args.lower.resolve()),
        "upper_certificate": str(args.upper.resolve()),
        "input_sha256": {"lower": digest(args.lower), "upper": digest(args.upper)},
        "fixed_plan": str(lower_plan),
        "fixed_plan_sha256": digest(lower_plan),
        "validation": {
            "partition_checks": partition_checks,
            "tolerance": 1e-8,
        },
        "by_missile": by_missile,
        "objectives": {
            "J_sum": {"lower": sum_lower, "upper": sum_upper},
            "J_min": {"lower": min_lower, "upper": min_upper},
            "J_all": {
                "lower_intervals": all_lower,
                "upper_intervals": all_upper,
                "lower": measure(all_lower),
                "upper": measure(all_upper),
            },
        },
    }
    for bounds in payload["objectives"].values():
        if float(bounds["lower"]) > float(bounds["upper"]) + 1e-9:
            raise RuntimeError(f"invalid aggregate bounds: {bounds}")
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
