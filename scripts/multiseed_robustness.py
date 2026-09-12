#!/usr/bin/env python3
"""Multi-seed robustness audit for the dimension-scaling experiment."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

import global_scaling_stress as g

N_BLOCKS = 10
SEED_STRIDE = 100000


def case_margins(d: int, p: int, seed: int) -> tuple[float, float]:
    state_rng = np.random.default_rng(seed + 100 * d + p)
    theta_cloud = state_rng.uniform(g.THETA_LOW, g.THETA_HIGH, size=(g.N_STATES, d))
    pair_mask = g.separated_pair_mask(theta_cloud, g.DELTA)
    shared = 2 * d + p + 1
    naive = 2 * d + 2 * p + 1
    out: dict[int, float] = {}
    for m in (shared, naive):
        design_rng = np.random.default_rng(seed + 10000 * d + 100 * p + m)
        margins = np.empty(g.N_DESIGNS, dtype=float)
        for trial in range(g.N_DESIGNS):
            z = np.sort(design_rng.uniform(g.Z_LOW, g.Z_HIGH, size=m))
            u_perp = g.nuisance_complement(z, p)
            embedding = g.model_values(theta_cloud, z) @ u_perp
            margins[trial] = g.finite_cloud_margin(embedding, pair_mask, m)
        out[m] = float(np.median(margins))
    return out[shared], out[naive]


def main() -> None:
    rows: list[dict[str, float | int]] = []
    for d, p in g.CASES:
        ratios = []
        for block in range(N_BLOCKS):
            seed = g.SEED + block * SEED_STRIDE
            shared, naive = case_margins(d, p, seed)
            ratio = shared / naive
            ratios.append(ratio)
            rows.append({
                "d": d,
                "p": p,
                "block": block,
                "seed": seed,
                "median_margin_shared": shared,
                "median_margin_naive": naive,
                "ratio_shared_to_naive": ratio,
            })
        print(
            f"(d,p)=({d},{p}): median ratio={np.median(ratios):.6f}, "
            f"range=[{np.min(ratios):.6f}, {np.max(ratios):.6f}]"
        )

    out = Path("results/multiseed_robustness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "d", "p", "block", "seed", "median_margin_shared",
        "median_margin_naive", "ratio_shared_to_naive",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "median_margin_shared": f"{float(row['median_margin_shared']):.10f}",
                "median_margin_naive": f"{float(row['median_margin_naive']):.10f}",
                "ratio_shared_to_naive": f"{float(row['ratio_shared_to_naive']):.10f}",
            })


if __name__ == "__main__":
    main()
