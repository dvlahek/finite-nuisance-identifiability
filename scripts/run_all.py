#!/usr/bin/env python3
"""Run every numerical experiment and regression check used by the manuscript."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> None:
    run("monte_carlo_spectroscopy.py")
    run("global_scaling_stress.py")
    run("multiseed_robustness.py")
    run("network_augmentation.py")
    run("network_theorem_regression.py")
    run("nonminimality_example.py")


if __name__ == "__main__":
    main()
