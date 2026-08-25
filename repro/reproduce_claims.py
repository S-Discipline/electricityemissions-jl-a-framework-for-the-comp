"""
Reproduction driver for

    Identification of Analytic Nonlinear Dynamical Systems with
    Non-asymptotic Guarantees   (NeurIPS 2024, arXiv:2411.0656)

The single entrypoint run by every experiment node. It reproduces the paper's
two headline numerical claims:

  (1) Least-squares estimation (LSE) error  ||theta_hat_T - theta*|| / ||theta*||
      decays approximately as O(1/sqrt(T)) in trajectory length T (Figure 1).

  (2) Set-membership estimation (SME) uncertainty-set diameter decays
      approximately as O(1/T) in trajectory length T (Figure 2).

Which system / noise type runs is controlled by the committed code on each
experiment branch (SCENARIO below), so the run command stays identical on every
node (fixed-contract rule). The empirical rate is measured as the slope of the
log-log error-vs-T curve in its last decade and compared against the theory's
O(1/sqrt T) (slope -1/2) and O(1/T) (slope -1) predictions.

Results are written to results/ as CSV + PNG and a compact JSON summary is
printed to stdout as the run's evidence.
"""

import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics import Pendulum, Quadrotor
from estimators import (lse_pendulum, lse_quadrotor, run_sme,
                        uncertainty_diameter)

SCENARIO = "pendulum_trunc_guass"   # <-- select the committed scenario per branch

# ---------- trajectory grids matching the paper's notebooks -------------
def _lse_grid(max_T):
    g = [1, 500]
    g.extend(range(1000, min(10000, max_T + 1), 1000))          # 1k..9k
    g.extend(range(10000, min(max_T, 200000) + 1, 10000))       # 10k..
    if g[-1] < max_T:
        g.append(max_T)
    return sorted(set(g))


def _sme_grid(max_T):
    g = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    g.extend(range(1000, min(10000, max_T + 1), 1000))
    g.extend(range(10000, min(max_T, 30000) + 1, 5000))
    if g[-1] < max_T:
        g.append(max_T)
    return sorted(set(g))


SCENARIOS = {
    # ---- pendulum ----
    "pendulum_uniform": dict(
        system="pendulum", distr="uniform",
        param_input=[-1.0, 1.0], param_dist=[-1.0, 1.0],
        # paper: LSE uses mean/std-truncgauss fields; for uniform param=[lb,ub]
        n_epochs_lse=20, n_epochs_sme=10,
        lse_grid=_lse_grid(100000), sme_grid=_sme_grid(30000),
        w_max=0.01, k=2.0),
    "pendulum_trunc_guass": dict(
        system="pendulum", distr="trunc_guass",
        param_input=[0.0, 1.0, 1], param_dist=[0.0, 0.1, 10],
        n_epochs_lse=20, n_epochs_sme=10,
        lse_grid=_lse_grid(100000), sme_grid=_sme_grid(30000),
        w_max=0.01, k=2.0),
    # ---- quadrotor ----
    "quadrotor_uniform": dict(
        system="quadrotor", distr="uniform",
        param_input=[-1.0, 1.0], param_dist=[-1.0, 1.0],
        n_epochs_lse=20, n_epochs_sme=10,
        lse_grid=_lse_grid(30002), sme_grid=_sme_grid(30000),
        w_max=0.01),
    "quadrotor_trunc_guass": dict(
        system="quadrotor", distr="trunc_guass",
        param_input=[0.0, 0.1, 1], param_dist=[0.0, 0.1, 1],
        n_epochs_lse=20, n_epochs_sme=10,
        lse_grid=_lse_grid(30002), sme_grid=_sme_grid(30000),
        w_max=0.01),
}

# ---------------------------------------------------------------------------
# data generation (paper's trajectory sampling, fixed seeds)
# ---------------------------------------------------------------------------
def gen_trajectories(system, distr, param_input, param_dist, max_T,
                     n_epochs, mult_u=None, mult_w=None, k=2.0):
    """Generate n_epochs trajectories. Returns list of (Delta_S, Phi)."""
    trajs = []
    seeds_u = range(100, 300)
    seeds_w = range(300, 500)
    for e in range(n_epochs):
        if system == "pendulum":
            mdl = Pendulum(k=k)
            Delta, Phi = mdl.trajectory(max_T, distr, seeds_u[e], seeds_w[e],
                                        param_input, param_dist,
                                        mult_u or (1.0,), mult_w or (1.0, 1.0))
        else:
            mdl = Quadrotor()
            Delta, Phi = mdl.trajectory(max_T, distr, seeds_u[e], seeds_w[e],
                                        param_input, param_dist,
                                        mult_u or (1.0, 0.2, 0.2, 0.2))
        trajs.append((Delta, Phi))
    return trajs


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def estimate_lse(system, Delta, Phi):
    if system == "pendulum":
        return lse_pendulum(Delta, Phi)
    return lse_quadrotor(Delta, Phi)


def estimate_sme(system, Delta, Phi, w_max, ground_truth, dim):
    pts = run_sme(Delta, Phi, w_max, dim, ground_truth)
    return uncertainty_diameter(pts)


# ---------------------------------------------------------------------------
# rate analysis: slope of log(error) vs log(T) in the last decade
# ---------------------------------------------------------------------------
def empirical_slope(T_all, err_all):
    T = np.asarray(T_all, float)
    err = np.asarray(err_all, float)
    # use points with T in the upper half-decade range to estimate asymptotic slope
    mask = T >= 0.316 * T.max()
    x = np.log(T[mask]); y = np.log(np.maximum(err[mask], 1e-15))
    if len(x) < 2:
        return float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, _), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(slope)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_scenario(cfg):
    system = cfg["system"]
    distr = cfg["distr"]
    outdir = f"results_{system}_{distr}"
    os.makedirs(outdir, exist_ok=True)

    ground_truth = Pendulum().theta_star if system == "pendulum" else Quadrotor().theta_star_vec
    dim = 2 if system == "pendulum" else 10
    norm_c = np.linalg.norm(ground_truth)

    t0 = time.time()

    # ---------- LSE ----------
    n_lse = cfg["n_epochs_lse"]
    lse_grid = cfg["lse_grid"]
    print(f"[LSE] scenario={SCENARIO} trajs={n_lse} grid={len(lse_grid)} T<= {max(lse_grid)}", flush=True)
    trajs = gen_trajectories(system, distr, cfg["param_input"], cfg["param_dist"],
                             max(lse_grid), n_lse, k=cfg.get("k", 2.0))
    mean_err, std_err = [], []
    for T in lse_grid:
        errs = []
        for e in range(n_lse):
            Delta, Phi = trajs[e]
            th = estimate_lse(system, Delta[:T], Phi[:T])
            errs.append(np.linalg.norm(th - ground_truth) / norm_c)
        mean_err.append(float(np.mean(errs)))
        std_err.append(float(np.std(errs)))
    slope_lse = empirical_slope(lse_grid, mean_err)
    print(f"[LSE] done T grid. empirical slope = {slope_lse:.3f} (theory -0.5)",
          f"final normalized error = {mean_err[-1]:.4e} (theory O(1/sqrtT))", flush=True)
    np.savetxt(os.path.join(outdir, "lse_empirical.csv"),
               np.column_stack([lse_grid, mean_err, std_err]), delimiter=",",
               header="T,mean_norm_err,std_norm_err")

    # ---------- SME ----------
    n_sme = cfg["n_epochs_sme"]
    sme_grid = cfg["sme_grid"]
    print(f"[SME] scenario={SCENARIO} trajs={n_sme} grid={len(sme_grid)} T<= {max(sme_grid)}", flush=True)
    # reuse trajectories (same seeds); SME uses the longer trajectory
    trajs_sme = gen_trajectories(system, distr, cfg["param_input"], cfg["param_dist"],
                                 max(sme_grid), n_sme, k=cfg.get("k", 2.0))
    mean_diam, std_diam = [], []
    for T in sme_grid:
        diams = []
        for e in range(n_sme):
            Delta, Phi = trajs_sme[e]
            d = estimate_sme(system, Delta[:T], Phi[:T], cfg["w_max"], ground_truth, dim)
            diams.append(d)
        mean_diam.append(float(np.mean(diams)))
        std_diam.append(float(np.std(diams)))
    slope_sme = empirical_slope(sme_grid, mean_diam)
    print(f"[SME] done. empirical slope = {slope_sme:.3f} (theory -1)",
          f"final normalized diameter = {mean_diam[-1]/norm_c:.4e}", flush=True)
    np.savetxt(os.path.join(outdir, "sme_empirical.csv"),
               np.column_stack([sme_grid, mean_diam, std_diam]), delimiter=",",
               header="T,mean_diam,std_diam")

    elapsed = time.time() - t0
    print(f"[DONE] elapsed={elapsed:.1f}s")

    # ---------- summary block (the run's evidence) ----------
    summary = {
        "scenario": SCENARIO,
        "system": system,
        "noise": distr,
        "claims": {
            "lse": {
                "paper_rate_claim": "O(1/sqrt(T))  (slope ~ -1/2)",
                "observed_slope": slope_lse,
                "final_norm_err": mean_err[-1],
                "assessment": _assess_slope(slope_lse, -0.5),
            },
            "sme": {
                "paper_rate_claim": "O(1/T)  (slope ~ -1)",
                "observed_slope": slope_sme,
                "final_norm_diam": mean_diam[-1] / norm_c,
                "assessment": _assess_slope(slope_sme, -1.0),
            },
        },
    }
    print("\n===== REPRODUCTION SUMMARY (JSON) =====")
    print(json.dumps(summary, indent=2))
    print("===== END SUMMARY =====")
    return summary


def _assess_slope(obs, theory):
    if np.isnan(obs):
        return "inconclusive: insufficient points"
    # within 0.1 of the theoretical slope => aligned
    if abs(obs - theory) <= abs(theory) * 0.3:
        return "aligned"
    if obs > theory + abs(theory) * 0.3:
        return "slower-than-claimed decay"
    return "faster-than-claimed decay"


if __name__ == "__main__":
    if SCENARIO not in SCENARIOS:
        sys.exit(f"unknown SCENARIO {SCENARIO}; options: {list(SCENARIOS)}")
    run_scenario(SCENARIOS[SCENARIO])
