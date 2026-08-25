# Reproduction — *Identification of Analytic Nonlinear Dynamical Systems with Non-asymptotic Guarantees* (arXiv:2411.0656)

> **Claim tested:** For a *linearly parameterized nonlinear* system `x_{t+1} = θ*·φ(x_t,u_t) + w_t`
> whose feature functions `φ` are **real-analytic**, least-squares (LSE) estimation error decays
> like **O(1/√T)** and set-membership (SME) uncertainty-set diameter decays like **O(1/T)** in
> trajectory length `T`, under passive i.i.d. exploration noise.
>
> **What we did:** Reproduced the paper's pendulum and quadrotor system-identification experiments
> (Section 4, Figures 1–2) on the user's Vast.ai compute (`ssh1.vast.ai:21062`, alias `eei-a10`),
> re-implementing the paper's official code faithfully. We swept trajectory length `T`, averaged the
> LSE error and SME diameter over the paper's trial counts, and measured the log–log decay slope in
> the last decade of `T`.
>
> **Assessment:**
> * **SME (O(1/T)) — reproduced** on **all four** panels (pendulum/quadrotor × uniform/trunc-Gaussian): observed log–log slopes −1.11 to −1.21, all within noise of the claimed −1.0.
> * **LSE (O(1/√T)) — reproduced on three of four** panels (both pendulums, quadrotor·uniform): slopes −0.39 to −0.61, near the claimed −0.5. The **quadrotor · truncated-Gaussian** LSE panel did **not** show the claimed decay in this setup (slope ≈ −0.05; its error plateaued at ≈1.9×10⁻³ — about 10× above the paper's published value). This run therefore did **not** show the reported effect on that single panel; we do not conclude the claim is wrong, only that it was not reproduced under our setup.
>
> **Paper vs observed (key numbers):**

| Panel (system · noise) | Method | Paper rate | Observed slope | Observed final error/diam |
|---|---|---|---|---|
| Pendulum · Uniform | LSE | O(1/√T) | −0.61 | 6.8×10⁻⁵ |
| Pendulum · Trunc-Gaussian | LSE | O(1/√T) | −0.61 | 2.1×10⁻⁵ |
| Quadrotor · Uniform | LSE | O(1/√T) | −0.39 | 2.2×10⁻³ |
| Quadrotor · Trunc-Gaussian | LSE | O(1/√T) | **−0.05** | **1.9×10⁻³** |
| Pendulum · Uniform | SME | O(1/T) | −1.21 | 3.0×10⁻⁴ |
| Pendulum · Trunc-Gaussian | SME | O(1/T) | −1.14 | 2.0×10⁻³ |
| Quadrotor · Uniform | SME | O(1/T) | −1.15 | 2.4×10⁻³ |
| Quadrotor · Trunc-Gaussian | SME | O(1/T) | −1.11 | 1.4×10⁻² |

> **Downscaling / substitutions:** Pendulum LSE was swept to *T* = 10⁵ (paper: 2×10⁵); all other
> grids match the paper. SME uses `scipy.optimize.linprog` (Chebyshev center / halfspace
> intersection) for the same LP the paper solves with cvxopt. All other dynamics/config follow the
> paper's source.
>
> **Compute:** runs executed on the user's 96-core Vast.ai box (`eei-a10`) via `orx` (SSH backend).
> CPU-only; no GPU. Pendulum runs ≈ 0.5 min, quadrotor runs ≈ 18–23 min each.
>
> **Reports & artifacts:**
> * [Full illustrated reproduction report](reports/repro-2411.0656/report.md) (figures + per-claim breakdown)
> * [Interactive marimo notebook](notebooks/repro-2411.0656.py) — the already-computed evidence, opens with the SME result; `pip install marimo && marimo edit notebooks/repro-2411.0656.py` (or `marimo run`)
> * Reproduction code in [`repro/`](repro/) and raw per-panel result CSVs under [`repro/results_*/`](repro/)

## Experiment log (provenance)

Every node runs the **identical** command — `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` — and differs only in the committed `SCENARIO` (the fixed-run-contract rule). `main` itself was **not run as an experiment** (publication surface only).

| Branch (experiment) | Change / scenario | Exact run command | Assessment (LSE · SME) | Compute |
|---|---|---|---|---|
| `orx/pendulum-lse-sme-truncated-gaussian` | pendulum, truncated-Gaussian (baseline) | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.61 ✓ · SME −1.14 ✓ | `eei-a10` ssh, ~46 s |
| `orx/pendulum-lse-sme-uniform` | pendulum, uniform | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.61 ✓ · SME −1.21 ✓ | `eei-a10` ssh, ~40 s |
| `orx/quadrotor-lse-sme-truncated-gaussian` | quadrotor, truncated-Gaussian | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.05 ⚠ · SME −1.11 ✓ | `eei-a10` ssh, ~18 min |
| `orx/quadrotor-lse-sme-uniform` | quadrotor, uniform | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.39 ✓ · SME −1.15 ✓ | `eei-a10` ssh, ~23 min |

*Assessment key: ✓ = within ~30% of the claimed slope (aligned); ⚠ = far from the claim (did not show the reported decay).*

---

## About this repository

_(Upstream project description continues below. This repository is the OpenResearch reproduction
project for the paper above; its historical `ElectricityEmissions.jl` name is unrelated to the
reproduction.)_
