# Finite Nuisance Identifiability

Reproducibility repository for the manuscript **"Finite Random Measurements for Analytic Inverse Problems with Shared Nuisance Structure"**.

The paper studies exact physical identifiability when finite scalar measurements contain a shared finite-dimensional nuisance contribution. Its main global result uses the dimension of the **relative nuisance family** rather than duplicating the nuisance dimension for two candidate states. For self-identification of a `d`-dimensional analytic model with a common `p`-dimensional additive nuisance space, the resulting universal almost-sure sufficient count is

\[
M \ge 2d+p+1.
\]

The repository reproduces the numerical illustrations used to separate this global analytic statement from local differential rank and to verify the finite-network augmentation example.

## Repository contents

- `scripts/monte_carlo_spectroscopy.py` — 5000-trial Monte Carlo experiment for the analytic spectroscopy example.
- `scripts/network_augmentation.py` — exact finite-network ambiguity and one-sensor augmentation example.
- `scripts/run_all.py` — reproduces every committed numerical result.
- `results/spectroscopy_conditioning.csv` — conditioning statistics reported in the manuscript.
- `results/network_augmentation.json` — ranks, residual response, and augmented singular values.
- `figures/spectroscopy_conditioning.svg` — visualization of local projected conditioning versus the global sufficient measurement count.
- `.github/workflows/reproduce.yml` — CI workflow that reruns the calculations on every push and pull request.

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

The Monte Carlo calculation is deterministic with seed `20260912`.

## Expected spectroscopy results

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
