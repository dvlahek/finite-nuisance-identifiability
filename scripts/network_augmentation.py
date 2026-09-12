#!/usr/bin/env python3
"""Reproduce the finite-network augmentation example from the manuscript."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUTPUT_DECIMALS = 12


def stable_float(value: float) -> float:
    """Round output-only values to suppress platform-level last-bit SVD drift."""
    return round(float(value), OUTPUT_DECIMALS)


def projector_orthogonal_to(h: np.ndarray) -> np.ndarray:
    return np.eye(h.shape[0]) - h @ np.linalg.pinv(h)


def compute() -> dict[str, object]:
    q = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    h = np.ones((3, 1), dtype=float)
    v = np.array([-1.0, 0.0, 1.0])

    p = projector_orthogonal_to(h)
    old_projected = p @ q
    old_rank = int(np.linalg.matrix_rank(old_projected, tol=1.0e-12))
    k = int(q.shape[1] - old_rank)

    q_plus = np.array([[0.0, 0.0, 1.0]])
    h_plus = np.array([[0.0]])
    r_plus = q_plus - h_plus @ np.linalg.pinv(h) @ q

    q_aug = np.vstack((q, q_plus))
    h_aug = np.vstack((h, h_plus))
    p_aug = projector_orthogonal_to(h_aug)
    augmented_singular_values = np.linalg.svd(p_aug @ q_aug, compute_uv=False)

    basis_k = v / np.linalg.norm(v)
    residual_margin_on_k = float(np.linalg.norm(r_plus @ basis_k))

    assert np.allclose(q @ v, h[:, 0])
    assert old_rank == 2 and k == 1
    assert not np.allclose(r_plus @ v, 0.0)
    assert np.linalg.matrix_rank(p_aug @ q_aug, tol=1.0e-12) == 3

    return {
        "Q": q.tolist(),
        "H": h.tolist(),
        "ambiguity_direction_v": v.tolist(),
        "projected_old_rank": old_rank,
        "ambiguity_dimension_k": k,
        "Q_plus": q_plus.tolist(),
        "H_plus": h_plus.tolist(),
        "R_plus": r_plus.tolist(),
        "R_plus_v": stable_float((r_plus @ v)[0]),
        "residual_margin_on_unit_K_basis": stable_float(residual_margin_on_k),
        "augmented_projected_singular_values": [stable_float(x) for x in augmented_singular_values],
        "augmented_min_singular_value": stable_float(augmented_singular_values[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/network_augmentation.json"))
    args = parser.parse_args()

    result = compute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"old projected rank = {result['projected_old_rank']}")
    print(f"ambiguity dimension k = {result['ambiguity_dimension_k']}")
    print(f"R_plus v = {result['R_plus_v']:.6f}")
    print("augmented singular values = " + ", ".join(f"{x:.7f}" for x in result["augmented_projected_singular_values"]))


if __name__ == "__main__":
    main()
