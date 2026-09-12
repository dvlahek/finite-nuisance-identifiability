#!/usr/bin/env python3
"""Reproduce the local-conditioning Monte Carlo experiment from the manuscript."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        writer = csv.writer(handle)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "finite-nuisance-identifiability"
    m = np.array([row["M"] for row in rows])
    med = np.array([row["median_sigma_min"] for row in rows])
    lo = np.array([row["p10_sigma_min"] for row in rows])
    hi = np.array([row["p90_sigma_min"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(m, med, marker="o", label="Median")
    ax.fill_between(m, lo, hi, alpha=0.2, label="10th--90th percentile")
    ax.axvline(7, linestyle="--", linewidth=1.2, label="Global sufficient bound M=7")
    ax.set_xlabel("Number of scalar measurements M")
    ax.set_ylabel("Smallest singular value of projected Jacobian")
    ax.set_xticks(m.astype(int))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)


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
