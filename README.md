# Finite Nuisance Identifiability

Reproducibility repository for the manuscript **"Finite Random Measurements for Analytic Inverse Problems with Shared Nuisance Structure"**.

The paper studies exact physical identifiability when finite scalar measurements contain a shared finite-dimensional nuisance contribution. Its main global result uses the dimension of the **relative nuisance family** rather than duplicating the nuisance dimension for two candidate states. For self-identification of a `d`-dimensional analytic model with a common `p`-dimensional additive nuisance space, the resulting universal almost-sure sufficient count is

\[
M \ge 2d+p+1.
\]

A direct application of a generic finite-measurement theorem to the augmented physical+nuisance state would instead give `2d+2p+1`. The shared-nuisance formulation therefore saves exactly `p` measurements in the universal sufficient count. The bound is sufficient and is not claimed to be model-specific minimal; an exact `d=p=1` counterexample in the manuscript and repository requires only two measurements although the universal count is four.

The repository reproduces the numerical illustrations used to separate this global analytic statement from local differential rank, stress-test the count across several `(d,p)` regimes, verify the finite-network augmentation example, and regression-test the explicit nonminimality example.

## Repository contents

- `scripts/monte_carlo_spectroscopy.py` — 5000-trial Monte Carlo experiment for the analytic spectroscopy example.
- `scripts/global_scaling_stress.py` — dimension-scaling stress test for `(d,p)=(1,1),(2,1),(2,2),(3,2)`.
- `scripts/network_augmentation.py` — exact finite-network ambiguity and one-sensor augmentation example.
- `scripts/nonminimality_example.py` — exact two-measurement counterexample showing that the universal count need not be minimal.
- `scripts/run_all.py` — reproduces every committed numerical result and exact regression check.
- `results/spectroscopy_conditioning.csv` — conditioning statistics reported in the manuscript.
- `results/global_scaling_stress.csv` — local conditioning and finite-cloud quotient-separation statistics.
- `results/network_augmentation.json` — ranks, residual response, and augmented singular values.
- `results/nonminimality_example.json` — deterministic output of the exact nonminimality example.
- `figures/spectroscopy_conditioning.svg` — local projected conditioning for the spectroscopy example.
- `figures/global_scaling_stress.svg` — measurement-count sweep for the scaling stress test.
- `.github/workflows/reproduce.yml` — CI workflow that reruns the calculations on every push and pull request and verifies the committed outputs.

## Reproduce

Python 3.12 is used in CI.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/run_all.py
```

All stochastic experiments use deterministic seeds derived from `20260912`.

## Analytic spectroscopy example

The physical model is

\[
f((a,b),z)=e^{az}+b e^{3z},
\]

with unknown affine nuisance baseline `alpha_0 + alpha_1 z`. The physical dimension is `d=2` and the nuisance dimension is `p=2`, so the theorem gives the global sufficient count `M=7`. Local nuisance-projected Jacobian rank is generically possible already at `M=4`.

| M | Full-rank fraction | Median sigma_min | 10th pct. | 90th pct. |
|---:|---:|---:|---:|---:|
| 3 | 0.000 | 0 | 0 | 0 |
| 4 | 1.000 | 0.01015 | 0.00065 | 0.05555 |
| 5 | 1.000 | 0.02897 | 0.00457 | 0.09372 |
| 6 | 1.000 | 0.04841 | 0.01175 | 0.12109 |
| 7 | 1.000 | 0.06582 | 0.02033 | 0.14431 |
| 8 | 1.000 | 0.08174 | 0.02923 | 0.16486 |

This does **not** claim global injectivity below `M=7`. It illustrates that local differential rank and global uniform almost-sure identifiability are different statements.

## Dimension-scaling stress test

The stress test uses the nonlinear analytic family

\[
f_\theta(z)=\sum_{j=1}^{d}\exp((1.5j+\theta_j)z),
\]

with `theta_j in [0.1,0.5]` and polynomial nuisance space `span{1,z,...,z^(p-1)}`. For each `(d,p)` pair, 180 physical states and 200 random measurement designs per `M` are used. The finite-cloud diagnostic only compares parameter pairs separated by at least `delta=0.12` and is normalized by `sqrt(M)`.

| d | p | Local `d+p` | Shared `2d+p+1` | Naive `2d+2p+1` | Saved | Median cloud margin at shared | At naive |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 4 | 5 | 1 | 0.04726 | 0.05462 |
| 2 | 1 | 3 | 6 | 7 | 1 | 0.00683 | 0.00736 |
| 2 | 2 | 4 | 7 | 9 | 2 | 0.00184 | 0.00271 |
| 3 | 2 | 5 | 9 | 11 | 2 | 0.00122 | 0.00131 |

At `M=d+p`, every tested design was locally full rank. At the shared global sufficient count, the median finite-cloud quotient margin retained approximately 68%--93% of the margin obtained at the larger naive augmented count. This is a numerical stress test, not a proof that the universal bound is sharp.

## Exact nonminimality example

For

\[
y_i=\theta z_i+\alpha,
\]

with `d=p=1`, the universal theorem gives `M>=4`. Two distinct sample points already give

\[
\theta=\frac{y_1-y_2}{z_1-z_2},\qquad \alpha=y_1-\theta z_1,
\]

so the exact almost-sure minimum for this model is `M=2`. One measurement is insufficient because any alternative `theta'` can be compensated by an adjusted constant nuisance. `scripts/nonminimality_example.py` verifies both statements and writes the deterministic reference output used by CI.

## Finite-network augmentation

For the manuscript example,

\[
Q=\begin{bmatrix}1&0&2\\0&1&1\\0&0&1\end{bmatrix},\qquad
H=\begin{bmatrix}1\\1\\1\end{bmatrix}.
\]

The nuisance-projected physical matrix has rank 2, leaving the one-dimensional ambiguity direction

\[
v=(-1,0,1)^T,
\]

because `Qv` lies in `range(H)`. The added measurement

\[
Q_+=\begin{bmatrix}0&0&1\end{bmatrix},\qquad H_+=0
\]

has nonzero residual response on this ambiguity. The augmented nuisance-projected physical matrix has singular values approximately

`1.4619022, 0.7778619, 0.5077133`,

so one added scalar measurement is both necessary and sufficient in this example.

## Reproducibility note

No external data sets are used. All numerical values are generated from analytic models specified in the manuscript and in the scripts in this repository.

## Author

Dino Vlahek  
Faculty of Organization and Informatics, University of Zagreb  
ORCID: 0000-0002-3911-8685
