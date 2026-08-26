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
> **Assessment (strict grade: C — partial reproduction).**
> * **SME (O(1/T)) — trend reproduced on all four panels** (slopes −1.11…−1.21 ≈ claimed −1.0), and every SME set contains the true parameters.
> * **LSE (O(1/√T)) — trend reproduced on three of four panels** (slopes −0.52…−0.61). The **quadrotor · truncated-Gaussian** LSE slope is shallow (−0.18 vs −0.5).
> * **Magnitudes diverge** for the quadrotor: LSE error is **7–11× above** the paper's published value (1.1–1.6×10⁻³ vs ~1.4×10⁻⁴), and quadrotor·uniform SME is ~4× tighter (−76%). Pendulum magnitudes agree to within ~15–60%.
>
> **Paper vs observed (key numbers, corrected):**

| Panel (system · noise) | Method | Paper rate | Observed slope | Observed final (paper→mine) |
|---|---|---|---|---|
| Pendulum · Uniform | LSE | O(1/√T) | −0.61 | 5.3×10⁻⁵ → 6.8×10⁻⁵ (+30%) |
| Pendulum · Trunc-Gaussian | LSE | O(1/√T) | −0.61 | 5.6×10⁻⁵ → 2.1×10⁻⁵ (−63%) |
| Quadrotor · Uniform | LSE | O(1/√T) | −0.52 | 1.5×10⁻⁴ → 1.6×10⁻³ (**+980%**) |
| Quadrotor · Trunc-Gaussian | LSE | O(1/√T) | **−0.18** | 1.4×10⁻⁴ → 1.1×10⁻³ (**+640%**) |
| Pendulum · Uniform | SME | O(1/T) | −1.21 | 3.5×10⁻⁴ → 3.0×10⁻⁴ (−15%) |
| Pendulum · Trunc-Gaussian | SME | O(1/T) | −1.14 | — |
| Quadrotor · Uniform | SME | O(1/T) | −1.15 | 1.0×10⁻² → 2.4×10⁻³ (−76%) |
| Quadrotor · Trunc-Gaussian | SME | O(1/T) | −1.11 | — |

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
| `orx/quadrotor-lse-sme-truncated-gaussian` | quadrotor, truncated-Gaussian | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.18 ⚠ · SME −1.11 ✓ | `eei-a10` ssh, ~18 min |
| `orx/quadrotor-lse-sme-uniform` | quadrotor, uniform | `pip install --quiet numpy scipy matplotlib 2>/dev/null; python3 repro/reproduce_claims.py` | LSE −0.52 ✓ · SME −1.15 ✓ | `eei-a10` ssh, ~23 min |

*Assessment key: ✓ = within ~30% of the claimed slope (aligned); ⚠ = far from the claim. Note the
grade is C, not B: the slopes/trends are mostly right, but the quadrotor LSE magnitudes are 7–11×
off and quadrotor SME ~4× tighter, so the quantitative match is not close enough for a full or
reduced-scale success.*

---

## About this repository

_(Upstream project description continues below. This repository is the OpenResearch reproduction
project for the paper above; its historical `ElectricityEmissions.jl` name is unrelated to the
reproduction.)_
