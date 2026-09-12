# Finite Nuisance Identifiability

Reproducibility repository for **Finite Random Measurements for Analytic Inverse Problems with Shared Nuisance Structure**.

The paper studies physical identifiability when finite scalar measurements contain a shared finite-dimensional nuisance contribution. For a `d`-dimensional analytic model with a common `p`-dimensional additive nuisance space, the main self-identification result gives

\[
M \ge 2d+p+1.
\]

Direct augmentation of the physical and nuisance variables would give `2d+2p+1`. The reduction comes from the collision geometry: two candidate states depend on one relative nuisance vector, not two independent nuisance vectors.

## Repository contents

- `scripts/monte_carlo_spectroscopy.py` - 5000-trial spectroscopy conditioning experiment.
- `scripts/global_scaling_stress.py` - dimension-scaling experiment for `(d,p)=(1,1),(2,1),(2,2),(3,2)`.
- `scripts/multiseed_robustness.py` - ten-seed robustness audit of the shared/naive quotient-margin comparison.
- `scripts/network_augmentation.py` - exact finite-network ambiguity and one-sensor augmentation example.
- `scripts/network_theorem_regression.py` - 1000 randomized checks of the augmentation theorem against the direct augmented projected-rank condition.
- `scripts/nonminimality_example.py` - exact `d=p=1` example with model-specific minimum `M=2`.
- `scripts/run_all.py` - runs all numerical experiments and regression checks.
- `results/` - committed reference outputs.
- `figures/` - deterministic SVG figure sources.
- `.github/workflows/reproduce.yml` - CI workflow that regenerates the outputs and fails on modified or untracked files in `results/` or `figures/`.

## Reproduce

CI uses Python 3.12.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/run_all.py
```

The main seed is `20260912`. The scaling experiment uses separate random streams for measurement designs and local-Jacobian evaluation, so changes to one diagnostic do not change the other.

## Spectroscopy example

The model is

\[
f((a,b),z)=e^{az}+b e^{3z},
\]

with an unknown affine baseline. Here `d=2` and `p=2`, so the global sufficient count is `M=7`. Local projected Jacobian rank is already possible at `M=4`.

| M | Full-rank fraction | Median sigma_min | 10th pct. | 90th pct. |
|---:|---:|---:|---:|---:|
| 3 | 0.000 | 0 | 0 | 0 |
| 4 | 1.000 | 0.01015 | 0.00065 | 0.05555 |
| 5 | 1.000 | 0.02897 | 0.00457 | 0.09372 |
| 6 | 1.000 | 0.04841 | 0.01175 | 0.12109 |
| 7 | 1.000 | 0.06582 | 0.02033 | 0.14431 |
| 8 | 1.000 | 0.08174 | 0.02923 | 0.16486 |

## Dimension-scaling experiment

The nonlinear family is

\[
f_\theta(z)=\sum_{j=1}^{d}\exp((1.5j+\theta_j)z),
\]

with polynomial nuisance space `span{1,z,...,z^(p-1)}`. Each regime uses 180 physical states and 200 random measurement designs per `M`.

| d | p | Local `d+p` | Shared `2d+p+1` | Naive `2d+2p+1` | Saved | Margin at shared | Margin at naive |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 4 | 5 | 1 | 0.04470 | 0.05147 |
| 2 | 1 | 3 | 6 | 7 | 1 | 0.00694 | 0.00762 |
| 2 | 2 | 4 | 7 | 9 | 2 | 0.00195 | 0.00248 |
| 3 | 2 | 5 | 9 | 11 | 2 | 0.00105 | 0.00148 |

Across ten independent seed blocks, the median shared-to-naive margin ratios are approximately `0.902`, `0.908`, `0.773`, and `0.812` for the four regimes. The full block-level values are stored in `results/multiseed_robustness.csv`.

## Finite-network augmentation

For

\[
Q=\begin{bmatrix}1&0&2\\0&1&1\\0&0&1\end{bmatrix},\qquad
H=\begin{bmatrix}1\\1\\1\end{bmatrix},
\]

the nuisance-projected physical matrix has rank 2 and leaves the ambiguity direction

\[
v=(-1,0,1)^T.
\]

One added scalar measurement resolves the ambiguity, and the augmented projected singular values are approximately

`1.4619022, 0.7778619, 0.5077133`.

The randomized regression test generates 1000 finite networks and compares the theorem condition `rank(R_+ B_K)=k` with direct full-rank testing after augmentation. The committed run gives `1000/1000` agreement.

## Model-specific lower count

For

\[
y_i=\theta z_i+\alpha,
\]

with `d=p=1`, two distinct sample points recover both `theta` and `alpha`, while one measurement leaves a one-parameter ambiguity. Thus this model has exact almost-sure minimum `M=2`, below the uniform bound `M=4`.

## Figure generation

The repository generates deterministic SVG figure sources. The manuscript PDF figures are vector conversions of these SVG files; the numerical content is fully determined by the committed SVG and CSV outputs.

## Reproducibility

No external data sets are used. GitHub Actions reruns all experiments and regression checks on every push and pull request and verifies that `results/` and `figures/` remain unchanged.

## Author

Dino Vlahek  
Faculty of Organization and Informatics, University of Zagreb  
ORCID: 0000-0002-3911-8685
