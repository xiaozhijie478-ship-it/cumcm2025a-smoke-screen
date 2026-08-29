"""Rebuild the stitched Q3 upper bound from its saved evidence files."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


ORDERS = set(itertools.permutations((1, 2, 3)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("q3_full_domain_certificate_manifest.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    layer = manifest["layers"]["three_contributors"]
    base_path = args.manifest.with_name(layer["global_evidence"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    shards = [item.copy() for item in base["results"] if item["domain"] == "three"]
    assert len(shards) == base["shards"]
    headings = sorted(item["theta_range_deg"] for item in shards)
    assert abs(headings[0][0]) <= 1e-12
    assert abs(headings[-1][1] - 360.0) <= 1e-12
    assert all(
        abs(left[1] - right[0]) <= 1e-12
        for left, right in zip(headings, headings[1:])
    )

    replacements = []
    for refinement in layer["refined_heading_shards"]:
        evidence = [
            json.loads(args.manifest.with_name(name).read_text(encoding="utf-8"))
            for name in refinement["explosion_order_evidence"]
        ]
        assert {tuple(item["explosion_order"]) for item in evidence} == ORDERS
        upper = max(item["global_upper"] for item in evidence)
        assert abs(upper - refinement["raw_upper_seconds"]) <= 1e-12
        shard = next(
            item
            for item in shards
            if max(
                abs(a - b)
                for a, b in zip(item["theta_range_deg"], refinement["heading_deg"])
            )
            <= 1e-9
        )
        shard["global_upper"] = min(shard["global_upper"], upper)
        replacements.append({"heading_deg": refinement["heading_deg"], "upper": upper})

    rebuilt = max(item["global_upper"] for item in shards)
    assert abs(rebuilt - layer["raw_upper_seconds"]) <= 1e-12
    layer_upper = max(
        manifest["layers"]["one_contributor"]["upper_seconds"],
        manifest["layers"]["two_contributors"]["upper_seconds"],
        rebuilt,
    )
    assert abs(layer_upper - manifest["raw_complete_domain_upper_seconds"]) <= 1e-12
    reported = manifest["reported_outward_rounded_upper_seconds"]
    assert reported >= layer_upper
    assert reported - layer_upper <= 0.001 + 1e-12
    print(
        json.dumps(
            {
                "raw_complete_domain_upper_seconds": layer_upper,
                "reported_outward_rounded_upper_seconds": reported,
                "refined_heading_shards": replacements,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
