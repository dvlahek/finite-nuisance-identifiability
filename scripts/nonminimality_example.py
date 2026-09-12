#!/usr/bin/env python3
"""Exact two-measurement counterexample to universal-bound minimality.

For f(theta,z)=theta*z with an unknown constant nuisance alpha, the general
shared-nuisance theorem gives 2d+p+1=4 when d=p=1. Two distinct scalar
samples already recover theta and alpha exactly, while one sample leaves a
one-parameter ambiguity.
"""
from __future__ import annotations

import json
from pathlib import Path


def recover_two(z1: float, z2: float, y1: float, y2: float) -> tuple[float, float]:
    if z1 == z2:
        raise ValueError("The two sampling points must be distinct.")
    theta = (y1 - y2) / (z1 - z2)
    alpha = y1 - theta * z1
    return theta, alpha


def main() -> None:
    d, p = 1, 1
    universal_bound = 2 * d + p + 1

    theta_true = 1.75
    alpha_true = -0.40
    z1, z2 = -0.70, 0.90
    y1 = theta_true * z1 + alpha_true
    y2 = theta_true * z2 + alpha_true
    theta_hat, alpha_hat = recover_two(z1, z2, y1, y2)

    assert abs(theta_hat - theta_true) < 1e-12
    assert abs(alpha_hat - alpha_true) < 1e-12

    # One-sample ambiguity: any alternative theta can be compensated by alpha.
    theta_alt = -0.25
    alpha_alt = alpha_true + (theta_true - theta_alt) * z1
    assert abs(theta_alt * z1 + alpha_alt - y1) < 1e-12

    result = {
        "physical_dimension_d": d,
        "nuisance_dimension_p": p,
        "universal_sufficient_bound": universal_bound,
        "exact_minimum_for_this_model": 2,
        "sampling_points": [z1, z2],
        "true_theta": theta_true,
        "true_alpha": alpha_true,
        "recovered_theta": round(theta_hat, 12),
        "recovered_alpha": round(alpha_hat, 12),
        "one_sample_alternative_theta": theta_alt,
        "one_sample_compensating_alpha": round(alpha_alt, 12),
    }

    out = Path("results/nonminimality_example.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "nonminimality example: universal bound=4, exact model-specific minimum=2; "
        f"recovered theta={theta_hat:.12g}, alpha={alpha_hat:.12g}"
    )


if __name__ == "__main__":
    main()
