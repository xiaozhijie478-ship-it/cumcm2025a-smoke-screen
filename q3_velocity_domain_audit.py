"""Scan Q3's two-contributor and three-contributor heading shards."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import q3_upper_velocity_branch_bound as bounder


DOMAINS = {
    "two": ("two", (0, 1)),
    "three": ("full", (0, 1, 2)),
}


def scan(
    domains: list[str],
    shards: int,
    dt: float,
    fine_dt: float | None,
    refine_below: float,
    n_phi: int,
    target: float,
    max_nodes: int,
    center_cells: int,
    event_cells: int,
    fine_center_cells: int | None = None,
    fine_event_cells: int | None = None,
    cell_refine_below: float | None = None,
) -> dict[str, object]:
    results = []
    width = 360.0 / shards
    for domain in domains:
        preset, required = DOMAINS[domain]
        for index in range(shards):
            angle = (index * width, (index + 1) * width)
            started = time.perf_counter()
            result = bounder.branch_bound(
                dt,
                fine_dt,
                refine_below,
                n_phi,
                target,
                max_nodes,
                center_cells,
                preset,
                7.0,
                True,
                required,
                event_cells,
                tuple(math.radians(value) for value in angle),
                True,
                fine_center_cells=fine_center_cells,
                fine_event_cells=fine_event_cells,
                cell_refine_below=cell_refine_below,
            )
            results.append(
                {
                    "domain": domain,
                    "shard": index,
                    "angle_deg": angle,
                    "elapsed_seconds": time.perf_counter() - started,
                    **result,
                }
            )
            print(
                f"domain={domain},shard={index + 1}/{shards},angle={angle},"
                f"upper={result['global_upper']:.6f},open={result['open']}",
                flush=True,
            )

    summary = {}
    for domain in domains:
        selected = [item for item in results if item["domain"] == domain]
        open_items = [item for item in selected if not item["target_certified"]]
        summary[domain] = {
            "shards": len(selected),
            "certified_shards": len(selected) - len(open_items),
            "open_shards": len(open_items),
            "max_upper": max(item["global_upper"] for item in selected),
            "open_angles_deg": [item["angle_deg"] for item in open_items],
            "elapsed_seconds": sum(item["elapsed_seconds"] for item in selected),
        }
    return {
        "decomposition": {
            "one_contributor": "independent cumulative upper <= 7.0 s",
            "two_contributors": "generic ordered two-bomb domain",
            "three_contributors": "full domain with e1,e2,e3 <= COVER_END",
        },
        "shards": shards,
        "target": target,
        "max_nodes_per_shard": max_nodes,
        "summary": summary,
        "results": results,
    }


def self_test() -> None:
    assert set(DOMAINS) == {"two", "three"}
    assert DOMAINS["two"] == ("two", (0, 1))
    assert DOMAINS["three"] == ("full", (0, 1, 2))
    assert 7.0 < 8.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="two,three")
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fine-dt", type=float)
    parser.add_argument("--refine-below", type=float, default=10.0)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--target", type=float, default=8.0)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--center-cells", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--event-cells", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--fine-center-cells", type=int, choices=[2, 4, 8])
    parser.add_argument("--fine-event-cells", type=int, choices=[2, 4, 8])
    parser.add_argument("--cell-refine-below", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("q3_velocity_domain_audit.json"),
    )
    args = parser.parse_args()
    self_test()
    domains = [item for item in args.domains.split(",") if item]
    if not domains or any(item not in DOMAINS for item in domains):
        parser.error("--domains accepts two,three")
    if args.shards <= 0:
        parser.error("--shards must be positive")
    result = scan(
        domains,
        args.shards,
        args.dt,
        args.fine_dt,
        args.refine_below,
        args.n_phi,
        args.target,
        args.max_nodes,
        args.center_cells,
        args.event_cells,
        args.fine_center_cells,
        args.fine_event_cells,
        args.cell_refine_below,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
