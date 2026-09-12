#!/usr/bin/env python3
"""Randomized regression check for the finite-network augmentation theorem."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

SEED = 20260912
N_CASES = 1000
TOL = 1e-9


def projector(H: np.ndarray) -> np.ndarray:
    return np.eye(H.shape[0]) - H @ np.linalg.pinv(H)


def null_basis(A: np.ndarray) -> np.ndarray:
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > TOL))
    return vh[rank:].T


def main() -> None:
    rng = np.random.default_rng(SEED)
    mismatches = 0
    ambiguity_histogram: dict[int, int] = {}

    for _ in range(N_CASES):
        d = int(rng.integers(1, 7))
        p = int(rng.integers(1, 4))
        n = int(rng.integers(p + 1, p + d + 2))

        H = rng.normal(size=(n, p))
        while np.linalg.matrix_rank(H, tol=TOL) < p:
            H = rng.normal(size=(n, p))
        Q = rng.normal(size=(n, d))

        A = projector(H) @ Q
        B_K = null_basis(A)
        k = B_K.shape[1]
        ambiguity_histogram[k] = ambiguity_histogram.get(k, 0) + 1

        m = int(rng.integers(0, max(2, k + 2)))
        Q_plus = rng.normal(size=(m, d)) if m else np.empty((0, d))
        H_plus = rng.normal(size=(m, p)) if m else np.empty((0, p))
        R_plus = Q_plus - H_plus @ np.linalg.pinv(H) @ Q

        theorem_condition = True if k == 0 else np.linalg.matrix_rank(R_plus @ B_K, tol=TOL) == k
        H_aug = np.vstack((H, H_plus))
        Q_aug = np.vstack((Q, Q_plus))
        direct_condition = np.linalg.matrix_rank(projector(H_aug) @ Q_aug, tol=TOL) == d
        if theorem_condition != direct_condition:
            mismatches += 1

    assert mismatches == 0
    result = {
        "seed": SEED,
        "n_cases": N_CASES,
        "mismatches": mismatches,
        "ambiguity_dimension_histogram": {str(k): v for k, v in sorted(ambiguity_histogram.items())},
    }
    out = Path("results/network_theorem_regression.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"network theorem regression: {N_CASES - mismatches}/{N_CASES} agreements")


if __name__ == "__main__":
    main()
