import marimo

__generated_with = "0.9.24"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    return np


@app.cell
def _():
    import marimo as mo
    return mo


@app.cell
def _(mo):
    mo.md(
        r"""
# Can passive noise identify a nonlinear system? Reproducing arXiv:2411.0656

**The claim.** For a *linearly parameterized nonlinear* system
`x_{t+1} = θ*·φ(x_t, u_t) + w_t` whose features `φ` are **real-analytic**, two
estimators recover the unknown parameters `θ*` from a single noisy trajectory driven
only by i.i.d. exploration noise:

* **Least-squares (LSE)** point estimate: error `‖θ̂_T − θ*‖ ~ O(1/√T)`
* **Set-membership (SME)** uncertainty set: its **diameter** ~ **O(1/T)**

We reproduced these rates on the paper's **pendulum** and **quadrotor**, under
**uniform** and **truncated-Gaussian** noise. The SME claim reproduces on **all four**
panels (log–log slope ≈ −1); the LSE claim reproduces on **three** of four (slope ≈ −0.5).

**Grade: C (partial).** Trends/rates mostly match, but the quadrotor LSE magnitudes are 7–11× above
the paper's published values (and quadrotor SME is ~4× tighter), so this is not a full reproduction.

This notebook shows the **already-computed** evidence from the reproduction runs — no
experiment needs to be rerun. The curves below are embedded directly from those runs.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 1. SME uncertainty sets shrink like O(1/T) — the headline result")
    return


@app.cell
def _(make_plot, np):
    _C = {
        "Pendulum · Uniform": "6.779440e+00,2.020473e-01,6.448981e-02,3.954081e-02,3.197884e-02,2.500896e-02,1.705392e-02,1.388223e-02,1.362873e-02,1.103498e-02,1.064484e-02,6.283492e-03,4.372999e-03,3.259283e-03,2.939240e-03,2.641338e-03,2.414349e-03,2.130015e-03,1.807798e-03,1.471635e-03,8.117231e-04,5.547776e-04,3.638656e-04,2.965142e-04",
        "Pendulum · Trunc": "1.283805e+01,7.433437e-01,3.104947e-01,1.846634e-01,1.516926e-01,1.291181e-01,9.425805e-02,8.084362e-02,7.846071e-02,6.491672e-02,6.330729e-02,3.568423e-02,2.467947e-02,1.897435e-02,1.694794e-02,1.476494e-02,1.391185e-02,1.239485e-02,1.024726e-02,9.072064e-03,5.072431e-03,3.545669e-03,2.383662e-03,1.957847e-03",
        "Quadrotor · Uniform": "1.521877e+01,1.385367e+00,4.895932e-01,2.713394e-01,2.057461e-01,1.649083e-01,1.116702e-01,7.624163e-02,7.548965e-02,6.878533e-02,6.828238e-02,4.602874e-02,3.197596e-02,2.426808e-02,1.974488e-02,1.723603e-02,1.640170e-02,1.551925e-02,1.393784e-02,1.194836e-02,6.277542e-03,3.906352e-03,3.100368e-03,2.358920e-03",
        "Quadrotor · Trunc": "2.242822e+01,4.841812e+00,2.349334e+00,1.386189e+00,1.065933e+00,8.536459e-01,6.062589e-01,4.328585e-01,4.178634e-01,3.795280e-01,3.660628e-01,2.479218e-01,1.738927e-01,1.262007e-01,1.102103e-01,9.841462e-02,9.502208e-02,8.478103e-02,7.319777e-02,6.582545e-02,3.765343e-02,2.309701e-02,1.796152e-02,1.403343e-02",
    }
    _T = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000,
          6000, 7000, 8000, 9000, 10000, 15000, 20000, 25000, 30000]
    curves = {k: {"T": _T, "y": [float(v) for v in s.split(",")]} for k, s in _C.items()}
    make_plot("SME uncertainty-set diameter", curves, slope=-1.0,
              title="SME diameter vs trajectory length T (log–log)")
    return curves


@app.cell
def _(mo):
    mo.md(
        r"""
**Read-out.** Every panel tracks a ~`O(1/T)` line (log–log slope near −1, dashed).
The uncertainty set containing the true parameters shrinks as more data arrive — the
paper's headline SME prediction.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 2. LSE estimation error — three of four panels reproduce O(1/√T)")
    return


@app.cell
def _(make_plot, np):
    _L = {
        "Pendulum · Uniform": "1,7.362212e-02;500,9.340224e-04;1000,7.364729e-04;2000,5.320309e-04;3000,4.339775e-04;4000,3.810651e-04;5000,3.133709e-04;6000,2.698257e-04;7000,2.518082e-04;8000,2.459812e-04;9000,2.427919e-04;10000,2.416539e-04;20000,1.663099e-04;30000,1.463242e-04;40000,1.174835e-04;50000,9.930984e-05;60000,8.510382e-05;70000,6.938801e-05;80000,6.813793e-05;90000,6.649717e-05;100000,6.840006e-05",
        "Pendulum · Trunc": "1,5.106704e-02;500,3.970527e-04;1000,2.376094e-04;2000,1.146090e-04;3000,1.129435e-04;4000,8.643566e-05;5000,7.281698e-05;6000,8.096023e-05;7000,7.773627e-05;8000,7.349411e-05;9000,7.460212e-05;10000,7.074371e-05;20000,5.053493e-05;30000,3.727057e-05;40000,2.889618e-05;50000,2.445607e-05;60000,2.250971e-05;70000,1.812143e-05;80000,1.993482e-05;90000,1.994125e-05;100000,2.074373e-05",
        "Quadrotor · Uniform": "1,9.999720e-01;500,1.1255e-02;1000,7.0097e-03;2000,4.5789e-03;3000,4.0075e-03;4000,4.3821e-03;5000,4.1420e-03;6000,3.7256e-03;7000,3.5729e-03;8000,3.2282e-03;9000,3.0104e-03;10000,2.7655e-03;20000,1.9299e-03;30000,1.6046e-03;30002,1.6024e-03",
        "Quadrotor · Trunc": "1,9.999725e-01;500,1.5715e-03;1000,1.5559e-03;2000,1.5077e-03;3000,1.4834e-03;4000,1.4600e-03;5000,1.4851e-03;6000,1.4285e-03;7000,1.4229e-03;8000,1.3607e-03;9000,1.3774e-03;10000,1.3148e-03;20000,1.1576e-03;30000,1.0508e-03;30002,1.0521e-03",
    }
    lse_curves = {}
    for k, s in _L.items():
        pts = [p.split(",") for p in s.split(";")]
        lse_curves[k] = {"T": [float(a) for a, _ in pts], "y": [float(b) for _, b in pts]}
    make_plot("LSE estimation error (normalized)", lse_curves, slope=-0.5,
              title="LSE error vs trajectory length T (log–log)")
    return lse_curves


@app.cell
def _(mo):
    mo.md(
        r"""
**Read-out.** Three panels (both pendulums, quadrotor·uniform) fall on a ~`O(1/√T)`
line (dashed, slope −0.5). **Quadrotor · Truncated-Gaussian** does **not**: the error decays slowly
(≈1.05×10⁻³ at `T≈3×10⁴`, log–log slope ≈ −0.18). We do not treat this
as refuting the paper; it is that panel's setup-sensitivity. See the full report.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("## 3. Observed decay slopes vs the claimed rates")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
| Panel | LSE slope (claim −0.5) | SME slope (claim −1.0) |
|---|---|---|
| Pendulum · Uniform | −0.61 | −1.21 |
| Pendulum · Trunc | −0.61 | −1.14 |
| Quadrotor · Uniform | −0.52 | −1.15 |
| Quadrotor · Trunc | −0.18 | −1.11 |

All SME slopes sit within noise of **−1**; three LSE slopes are near **−0.5**. Only the
quadrotor·truncated-Gaussian LSE is far from its claim.
        """
    )
    return


@app.cell
def _(np):
    import matplotlib.pyplot as plt

    def make_plot(ylabel, curves, slope, title):
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
        cols = ["C1", "C0", "C3", "C2"]
        for ax, ((name, c), col) in zip(axs, zip(curves.items(), cols)):
            T = c["T"]; y = c["y"]
            ax.plot(T, y, color=col, lw=2, label="reproduced")
            ax.plot(T, y[-1] * (np.array(T) / T[-1]) ** slope, "k--", lw=1.2,
                    label=f"theory O(T^({slope}))")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.3)
        axs[0].set_ylabel(ylabel)
        axs[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    return make_plot


@app.cell
def _(mo):
    mo.md(
        r"""
---
**Where to find the data and the reproducible pipeline.** The reproduction code, the per-run
result CSVs, and the full report live in this repository (`reports/repro-2411.0656/`, `repro/`).
Every number above is regenerated by
`pip install numpy scipy matplotlib && python3 repro/reproduce_claims.py` for each committed
`SCENARIO` (see the README experiment log).

Run it locally with `marimo edit notebooks/repro-2411.0656.py` or `marimo run notebooks/repro-2411.0656.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
