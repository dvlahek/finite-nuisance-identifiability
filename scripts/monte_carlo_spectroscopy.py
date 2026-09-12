#!/usr/bin/env python3
"""Reproduce the local-conditioning Monte Carlo experiment from the manuscript."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SEED = 20260912
N_TRIALS = 5000
TOL = 1.0e-10
M_VALUES = tuple(range(3, 9))


def nuisance_projector(z: np.ndarray) -> np.ndarray:
    """Orthogonal projector away from the affine nuisance span {1, z}."""
    h = np.column_stack((np.ones_like(z), z))
    return np.eye(z.size) - h @ np.linalg.pinv(h)


def effective_jacobian(a: float, z: np.ndarray) -> np.ndarray:
    """Nuisance-projected Jacobian for f((a,b),z)=exp(a z)+b exp(3z)."""
    physical = np.column_stack((z * np.exp(a * z), np.exp(3.0 * z)))
    return nuisance_projector(z) @ physical


def run_experiment(seed: int = SEED, n_trials: int = N_TRIALS) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for m in M_VALUES:
        sigma_min = np.empty(n_trials, dtype=float)
        for trial in range(n_trials):
            z = rng.uniform(-1.0, 1.0, size=m)
            a = rng.uniform(0.2, 1.2)
            # b does not enter the Jacobian, but it is sampled to reproduce the
            # manuscript's physical-parameter draw exactly.
            _b = rng.uniform(0.5, 1.5)
            singular_values = np.linalg.svd(effective_jacobian(a, z), compute_uv=False)
            sigma_min[trial] = singular_values[-1]

        effective = sigma_min.copy()
        effective[effective <= TOL] = 0.0
        rows.append(
            {
                "M": float(m),
                "full_rank_fraction": float(np.mean(sigma_min > TOL)),
                "median_sigma_min": float(np.median(effective)),
                "p10_sigma_min": float(np.quantile(effective, 0.10)),
                "p90_sigma_min": float(np.quantile(effective, 0.90)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["M", "full_rank_fraction", "median_sigma_min", "p10_sigma_min", "p90_sigma_min"])
        for row in rows:
            writer.writerow(
                [
                    int(row["M"]),
                    f'{row["full_rank_fraction"]:.6f}',
                    f'{row["median_sigma_min"]:.8f}',
                    f'{row["p10_sigma_min"]:.8f}',
                    f'{row["p90_sigma_min"]:.8f}',
                ]
            )


def write_figure(rows: list[dict[str, float]], output: Path) -> None:
    """Write a compact dependency-free deterministic SVG figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 440
    left, right, top, bottom = 82, 24, 28, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = 0.18

    def x_map(m: float) -> float:
        return left + (m - 3.0) / 5.0 * plot_w

    def y_map(v: float) -> float:
        return top + (1.0 - v / y_max) * plot_h

    median = [(x_map(r["M"]), y_map(r["median_sigma_min"])) for r in rows]
    upper = [(x_map(r["M"]), y_map(r["p90_sigma_min"])) for r in rows]
    lower = [(x_map(r["M"]), y_map(r["p10_sigma_min"])) for r in reversed(rows)]
    band_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
    median_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in median)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="black"/>',
    ]

    for tick in (0.00, 0.05, 0.10, 0.15):
        y = y_map(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" font-size="13" text-anchor="end">{tick:.2f}</text>')

    for m in M_VALUES:
        x = x_map(float(m))
        parts.append(f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+6}" stroke="black"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+24}" font-size="13" text-anchor="middle">{m}</text>')

    x7 = x_map(7.0)
    parts.extend(
        [
            f'<polygon points="{band_points}" fill="#bdbdbd" fill-opacity="0.55" stroke="none"/>',
            f'<line x1="{x7:.2f}" y1="{top}" x2="{x7:.2f}" y2="{height-bottom}" stroke="#555555" stroke-width="2" stroke-dasharray="7,5"/>',
            f'<polyline points="{median_points}" fill="none" stroke="black" stroke-width="2.4"/>',
        ]
    )
    for x, y in median:
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="black" stroke-width="2"/>')

    parts.extend(
        [
            f'<text x="{left + plot_w/2:.2f}" y="{height-18}" font-size="15" text-anchor="middle">Number of scalar measurements M</text>',
            f'<text x="20" y="{top + plot_h/2:.2f}" font-size="15" text-anchor="middle" transform="rotate(-90 20 {top + plot_h/2:.2f})">Smallest singular value of projected Jacobian</text>',
            f'<text x="{x7+8:.2f}" y="{top+18}" font-size="12">global sufficient bound M=7</text>',
            '<line x1="100" y1="48" x2="132" y2="48" stroke="black" stroke-width="2.4"/>',
            '<circle cx="116" cy="48" r="4" fill="white" stroke="black" stroke-width="2"/>',
            '<text x="140" y="52" font-size="12">median</text>',
            '<rect x="100" y="61" width="32" height="12" fill="#bdbdbd" fill-opacity="0.55"/>',
            '<text x="140" y="72" font-size="12">10th–90th percentile</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/spectroscopy_conditioning.csv"))
    parser.add_argument("--figure", type=Path, default=Path("figures/spectroscopy_conditioning.svg"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    rows = run_experiment(seed=args.seed, n_trials=args.trials)
    write_csv(rows, args.csv)
    write_figure(rows, args.figure)

    for row in rows:
        print(
            f"M={int(row['M'])}: full-rank={row['full_rank_fraction']:.3f}, "
            f"median={row['median_sigma_min']:.5f}, "
            f"p10={row['p10_sigma_min']:.5f}, p90={row['p90_sigma_min']:.5f}"
        )


if __name__ == "__main__":
    main()
