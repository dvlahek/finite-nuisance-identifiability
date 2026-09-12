#!/usr/bin/env python3
"""Dimension-scaling stress test for shared-nuisance finite measurements.

The experiment is deliberately diagnostic rather than a proof of sharpness.
For several (d,p) pairs it samples analytic exponential-mixture models,
projects out a p-dimensional polynomial nuisance space, and records
(i) local projected-Jacobian conditioning and (ii) a finite-cloud proxy for
global separation among physically distinct parameter states.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np

SEED = 20260912
N_STATES = 180
N_DESIGNS = 200
DELTA = 0.12
THETA_LOW = 0.10
THETA_HIGH = 0.50
Z_LOW = -0.60
Z_HIGH = 0.60
CASES = ((1, 1), (2, 1), (2, 2), (3, 2))
TOL = 1.0e-10


def nuisance_complement(z: np.ndarray, p: int) -> np.ndarray:
    """Return an orthonormal basis for the complement of span{1,z,...,z^(p-1)}."""
    h = np.column_stack([z ** k for k in range(p)])
    u, _, _ = np.linalg.svd(h, full_matrices=True)
    return u[:, p:]


def model_values(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Analytic model f_theta(z)=sum_j exp((1.5(j+1)+theta_j) z)."""
    d = theta.shape[1]
    base = 1.5 * np.arange(1, d + 1, dtype=float)
    return np.exp((base[None, :, None] + theta[:, :, None]) * z[None, None, :]).sum(axis=1)


def physical_jacobian(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    d = theta.size
    base = 1.5 * np.arange(1, d + 1, dtype=float)
    return np.column_stack([z * np.exp((base[j] + theta[j]) * z) for j in range(d)])


def separated_pair_mask(theta: np.ndarray, delta: float) -> np.ndarray:
    diff = theta[:, None, :] - theta[None, :, :]
    distance = np.linalg.norm(diff, axis=2)
    return (distance >= delta) & np.triu(np.ones(distance.shape, dtype=bool), 1)


def finite_cloud_margin(embedding: np.ndarray, mask: np.ndarray, m: int) -> float:
    gram = embedding @ embedding.T
    diag = np.diag(gram)
    sq = np.maximum(diag[:, None] + diag[None, :] - 2.0 * gram, 0.0)
    return float(np.min(np.sqrt(sq)[mask]) / math.sqrt(m))


def run() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for d, p in CASES:
        state_rng = np.random.default_rng(SEED + 100 * d + p)
        theta_cloud = state_rng.uniform(THETA_LOW, THETA_HIGH, size=(N_STATES, d))
        pair_mask = separated_pair_mask(theta_cloud, DELTA)

        local_count = d + p
        shared_bound = 2 * d + p + 1
        naive_count = 2 * d + 2 * p + 1

        for m in range(local_count, naive_count + 1):
            design_rng = np.random.default_rng(SEED + 10000 * d + 100 * p + m)
            local_rng = np.random.default_rng(SEED + 20000 * d + 200 * p + m)
            margins = np.empty(N_DESIGNS, dtype=float)
            sigmas = np.empty(N_DESIGNS, dtype=float)

            for trial in range(N_DESIGNS):
                z = np.sort(design_rng.uniform(Z_LOW, Z_HIGH, size=m))
                u_perp = nuisance_complement(z, p)
                embedding = model_values(theta_cloud, z) @ u_perp
                margins[trial] = finite_cloud_margin(embedding, pair_mask, m)

                theta0 = local_rng.uniform(THETA_LOW, THETA_HIGH, size=d)
                j_eff = u_perp.T @ physical_jacobian(theta0, z)
                singular_values = np.linalg.svd(j_eff, compute_uv=False)
                sigmas[trial] = singular_values[-1] if singular_values.size >= d else 0.0

            rows.append(
                {
                    "d": d,
                    "p": p,
                    "M": m,
                    "local_rank_count": local_count,
                    "shared_bound": shared_bound,
                    "naive_augmented_count": naive_count,
                    "measurements_saved": naive_count - shared_bound,
                    "full_rank_fraction": float(np.mean(sigmas > TOL)),
                    "median_sigma_min": float(np.median(sigmas)),
                    "median_cloud_margin": float(np.median(margins)),
                    "p10_cloud_margin": float(np.quantile(margins, 0.10)),
                    "p90_cloud_margin": float(np.quantile(margins, 0.90)),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "d", "p", "M", "local_rank_count", "shared_bound", "naive_augmented_count",
        "measurements_saved", "full_rank_fraction", "median_sigma_min",
        "median_cloud_margin", "p10_cloud_margin", "p90_cloud_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("full_rank_fraction", "median_sigma_min", "median_cloud_margin", "p10_cloud_margin", "p90_cloud_margin"):
                out[key] = f"{float(out[key]):.10f}"
            writer.writerow(out)


def write_svg(rows: list[dict[str, float]], path: Path) -> None:
    """Write a deterministic, dependency-free log-scale SVG figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 470
    left, right, top, bottom = 82, 28, 28, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 2.0, 11.0
    y_min, y_max = 1e-5, 1e-1

    def X(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def Y(y: float) -> float:
        ly = math.log10(max(y, y_min))
        return top + (math.log10(y_max) - ly) / (math.log10(y_max) - math.log10(y_min)) * plot_h

    styles = [
        ("#1f77b4", ""),
        ("#d62728", "7,4"),
        ("#2ca02c", "3,3"),
        ("#9467bd", "10,3,2,3"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans,Arial,sans-serif" font-size="13" fill="black">',
    ]
    for y in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        yy = Y(y)
        svg.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#dddddd" stroke-width="1"/>')
        svg.append(f'<text x="{left-12}" y="{yy+4:.2f}" text-anchor="end">10^{int(round(math.log10(y)))}</text>')
    for x in range(2, 12):
        xx = X(x)
        svg.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+plot_h}" stroke="#f1f1f1" stroke-width="1"/>')
        svg.append(f'<text x="{xx:.2f}" y="{top+plot_h+24}" text-anchor="middle">{x}</text>')
    svg.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="1.2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="1.2"/>')
    svg.append(f'<text x="{left+plot_w/2:.2f}" y="{height-18}" text-anchor="middle">Number of scalar measurements M</text>')
    svg.append(f'<text transform="translate(22 {top+plot_h/2:.2f}) rotate(-90)" text-anchor="middle">Median finite-cloud quotient margin</text>')

    for idx, (d, p) in enumerate(CASES):
        case_rows = [r for r in rows if int(r["d"]) == d and int(r["p"]) == p]
        color, dash = styles[idx]
        points = " ".join(f'{X(float(r["M"])):.2f},{Y(float(r["median_cloud_margin"])):.2f}' for r in case_rows)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        for r in case_rows:
            x, y = X(float(r["M"])), Y(float(r["median_cloud_margin"]))
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.0" fill="{color}"/>')
            if int(r["M"]) == int(r["shared_bound"]):
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7.0" fill="none" stroke="{color}" stroke-width="2"/>')
            if int(r["M"]) == int(r["naive_augmented_count"]):
                svg.append(f'<rect x="{x-6:.2f}" y="{y-6:.2f}" width="12" height="12" fill="none" stroke="{color}" stroke-width="2"/>')

    lx, ly = 500, 42
    svg.append(f'<rect x="{lx-12}" y="{ly-20}" width="238" height="128" fill="white" stroke="#bbbbbb"/>')
    for idx, (d, p) in enumerate(CASES):
        color, dash = styles[idx]
        yy = ly + idx * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+36}" y2="{yy}" stroke="{color}" stroke-width="2.2"{dash_attr}/>')
        svg.append(f'<text x="{lx+45}" y="{yy+4}">(d,p)=({d},{p})</text>')
    yy = ly + 4 * 22 + 5
    svg.append(f'<circle cx="{lx+8}" cy="{yy}" r="6" fill="none" stroke="black" stroke-width="1.7"/><text x="{lx+23}" y="{yy+4}">shared bound</text>')
    svg.append(f'<rect x="{lx+2}" y="{yy+15}" width="12" height="12" fill="none" stroke="black" stroke-width="1.7"/><text x="{lx+23}" y="{yy+26}">naive augmented count</text>')
    svg.append('</g></svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    rows = run()
    write_csv(rows, Path("results/global_scaling_stress.csv"))
    write_svg(rows, Path("figures/global_scaling_stress.svg"))
    for d, p in CASES:
        shared = 2*d + p + 1
        naive = 2*d + 2*p + 1
        row_s = next(r for r in rows if r["d"] == d and r["p"] == p and r["M"] == shared)
        row_n = next(r for r in rows if r["d"] == d and r["p"] == p and r["M"] == naive)
        print(
            f"(d,p)=({d},{p}): local={d+p}, shared={shared}, naive={naive}, "
            f"saved={p}, margin@shared={row_s['median_cloud_margin']:.6g}, "
            f"margin@naive={row_n['median_cloud_margin']:.6g}"
        )


if __name__ == "__main__":
    main()
