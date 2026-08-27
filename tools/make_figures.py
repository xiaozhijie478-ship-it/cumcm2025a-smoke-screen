"""Generate paper-ready figures for Q3-Q5 from saved JSON evidence.

Outputs (into figures/):
  fig_q3_timeline.png        Q3 cumulative occlusion timeline (3 bombs, 1 UAV)
  fig_q4_timeline.png        Q4 cumulative occlusion timeline (3 UAVs)
  fig_q5_timeline.png        Q5 per-missile occlusion timeline (11 bombs)
  fig_q5_joint_additions.png Q5 certified joint-only coverage additions
  fig_q5_tradeoff.png        Q5 J_sum vs J_min / J_all candidate tradeoff
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "M1": "#1f77b4",
    "M2": "#ff7f0e",
    "M3": "#2ca02c",
    "joint": "#d62728",
    "baseline": "#9ecae1",
    "final": "#d62728",
}


def load(name: str):
    with open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return json.load(fh)


def draw_timeline(ax, rows: list[tuple[str, list[list[float]], str]], xlim: tuple[float, float]):
    for i, (label, intervals, color) in enumerate(rows):
        for a, b in intervals:
            ax.barh(i, b - a, left=a, height=0.55, color=color, edgecolor="black", linewidth=0.4)
        dur = sum(b - a for a, b in intervals)
        ax.text(xlim[1] + 0.3, i, f"{dur:.4f} s", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(xlim[0], xlim[1] + 3.2)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", alpha=0.3)


def fig_q3_timeline():
    bombs = [b for b in load("q5_bomb_table.json")["bombs"] if b["uav"] == "FY1"]
    rows = [
        (f"Bomb {b['number']}", b["individual_intervals"], COLORS["M1"])
        for b in sorted(bombs, key=lambda x: x["number"])
    ]
    total = sum(b["individual_duration"] for b in bombs)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    draw_timeline(ax, rows, (0, 16))
    ax.set_title(f"Q3: cumulative occlusion timeline, 3 bombs / 1 UAV (J3 = {total:.6f} s)")
    ax.set_yticks([0, 1, 2])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q3_timeline.png"), dpi=200)
    plt.close(fig)


def fig_q4_timeline():
    cert = load("q4_independent_upper_certificate.json")["certificates"]
    rows = [
        (fy, cert[fy]["current_intervals"], color)
        for fy, color in (("FY1", COLORS["M1"]), ("FY2", COLORS["M2"]), ("FY3", COLORS["M3"]))
    ]
    total = sum(cert[fy]["current_duration"] for fy in ("FY1", "FY2", "FY3"))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    draw_timeline(ax, rows, (0, 30))
    ax.set_title(f"Q4: cumulative occlusion timeline, 3 UAVs (J4 = {total:.6f} s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q4_timeline.png"), dpi=200)
    plt.close(fig)


def fig_q5_timeline():
    dense = load("q5_block_attacked_validation_v2.json")["resolutions"]["dense"]["metrics"]
    rows = [
        (m, dense["intervals"][m], COLORS[m])
        for m in ("M1", "M2", "M3")
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    draw_timeline(ax, rows, (0, 34))
    for a, b in dense["simultaneous_intervals"]:
        ax.axvspan(a, b, color=COLORS["joint"], alpha=0.18)
    ax.set_title(
        f"Q5: per-missile occlusion timeline (J_sum = {dense['total']:.6f} s, "
        f"J_min = {dense['minimum']:.6f} s, J_all = {dense['simultaneous']:.6f} s)"
    )
    ax.legend(
        handles=[
            Patch(color=COLORS["M1"], label="M1"),
            Patch(color=COLORS["M2"], label="M2"),
            Patch(color=COLORS["M3"], label="M3"),
            Patch(color=COLORS["joint"], alpha=0.4, label="simultaneous (J_all)"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q5_timeline.png"), dpi=200)
    plt.close(fig)


def fig_q5_joint_additions():
    hybrid = load("q5_joint_hybrid_certificate.json")["hybrid"]["by_missile"]
    rows, labels = [], []
    for m in ("M1", "M2", "M3"):
        single = hybrid[m]["single_union"]
        joint = hybrid[m]["joint_only_certificate"]["covered"]
        rows.append((f"{m} baseline", single, COLORS["baseline"]))
        rows.append((f"{m} + joint-only", joint, COLORS["joint"]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    draw_timeline(ax, rows, (0, 34))
    ax.set_title("Q5: certified joint-only coverage additions (multi-ball)")
    ax.legend(
        handles=[
            Patch(color=COLORS["baseline"], label="single-ball baseline"),
            Patch(color=COLORS["joint"], label="certified joint-only addition"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q5_joint_additions.png"), dpi=200)
    plt.close(fig)


def fig_q5_tradeoff():
    audit = load("q5_candidate_top_audit.json")["ranking"]
    xs = [e["metrics"]["total"] for e in audit]
    y_min = [e["metrics"]["minimum"] for e in audit]
    y_all = [e["metrics"]["simultaneous"] for e in audit]

    final = load("q5_block_attacked_validation_v2.json")["resolutions"]["dense"]["metrics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ys, ylabel in (
        (ax1, y_min, "J_min (s)"),
        (ax2, y_all, "J_all (s)"),
    ):
        ax.scatter(xs, ys, s=42, color=COLORS["M2"], zorder=3, label="audited candidates")
        ax.scatter(
            [final["total"]],
            [final["minimum"] if ylabel.startswith("J_min") else final["simultaneous"]],
            s=90,
            color=COLORS["final"],
            marker="*",
            zorder=4,
            label="final 11-bomb plan",
        )
        ax.set_xlabel("J_sum (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Q5: J_sum vs J_min / J_all candidate tradeoff")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_q5_tradeoff.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_q3_timeline()
    fig_q4_timeline()
    fig_q5_timeline()
    fig_q5_joint_additions()
    fig_q5_tradeoff()
    print("figures written to", OUT)
