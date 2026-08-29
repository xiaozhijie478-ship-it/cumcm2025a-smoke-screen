"""Merge per-missile Q5 lower certificates without changing their proof sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import q5_optimize as q5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not records:
        raise RuntimeError("at least one partial certificate is required")

    reference = records[0]
    invariant_fields = (
        "plan",
        "validation",
        "bomb_table",
        "time_tolerance",
        "joint_max_cells",
        "single_max_cells",
        "single_initial_chunk",
        "single_padding",
        "numerical_guard",
        "input_sha256",
    )
    for index, record in enumerate(records[1:], start=2):
        for field in invariant_fields:
            if record[field] != reference[field]:
                raise RuntimeError(
                    f"partial certificate {index} disagrees on {field}: "
                    f"{record[field]!r} != {reference[field]!r}"
                )

    source_hashes = reference["input_sha256"]
    source_paths = {
        "plan": Path(reference["plan"]),
        "validation": Path(reference["validation"]),
        "bomb_table": Path(reference["bomb_table"]),
    }
    for name, path in source_paths.items():
        if digest(path) != source_hashes[name]:
            raise RuntimeError(f"{name} changed after the partial certificates were made")

    by_missile: dict[str, object] = {}
    source_certificates = []
    for path, record in zip(args.inputs, records):
        source_certificates.append({"path": str(path.resolve()), "sha256": digest(path)})
        declared = set(record["missiles"])
        actual = set(record["hybrid"]["by_missile"])
        if declared != actual:
            raise RuntimeError(
                f"{path} declares {sorted(declared)} but contains {sorted(actual)}"
            )
        duplicate = actual.intersection(by_missile)
        if duplicate:
            raise RuntimeError(f"duplicate missile certificates: {sorted(duplicate)}")
        by_missile.update(record["hybrid"]["by_missile"])

    expected = set(q5.MISSILES)
    if set(by_missile) != expected:
        raise RuntimeError(
            f"merged certificate must contain {sorted(expected)}, got {sorted(by_missile)}"
        )

    payload = {field: reference[field] for field in invariant_fields}
    payload.update(
        {
            "scope": (
                "strict continuous-time lower certificate for one fixed plan; "
                "sampled intervals are candidate generators only"
            ),
            "missiles": list(q5.MISSILES),
            "source_certificates": source_certificates,
            "hybrid": {
                "by_missile": {name: by_missile[name] for name in q5.MISSILES},
                "total_certified_lower": sum(
                    float(by_missile[name]["certified_duration_lower"])
                    for name in q5.MISSILES
                ),
                "total_dense": sum(
                    float(by_missile[name]["dense_duration"])
                    for name in q5.MISSILES
                ),
            },
        }
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
