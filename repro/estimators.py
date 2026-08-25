"""
Estimators used in the reproduction:

    Identification of Analytic Nonlinear Dynamical Systems with
    Non-asymptotic Guarantees   (NeurIPS 2024, arXiv:2411.0656)

Implements the two identification methods studied in the paper:
  * least-squares estimation (LSE)  -- point estimate theta_hat
  * set-membership estimation (SME) -- uncertainty set containing theta*

The LSE routines mirror the paper's lse_pend.py / lse_qdt.py. The SME routines
compute the feasible uncertainty polytope
    { theta : |S_t - theta . phi_t| <= w_max,  t=1..T }
as an intersection of halfspaces (per component of the regression), then report
its diameter (max pairwise vertex distance). Feasibility is found with
scipy.optimize.linprog, the same operation the paper performs with cvxopt.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, distance_matrix


# ---------------------------------------------------------------------------
# Least-squares estimation
# ---------------------------------------------------------------------------
def lse_pendulum(Delta_S, Phi):
    """theta_hat for the pendulum (2 unknown scalars)."""
    Y = np.array(Delta_S).reshape(-1, 1)
    X = np.array(Phi)
    theta_hat = np.linalg.pinv(X) @ Y
    return np.array([theta_hat[0, 0], theta_hat[1, 0]])


def _quad_lse_matrices(Delta_S, Phi):
    """Assemble the LSE data matrices the way lse_qdt.py does."""
    Y, X = [], []
    for delta, phi in zip(Delta_S, Phi):
        Y.append([delta[0], 0., 0., 0.])
        Y.append([delta[1], 0., 0., 0.])
        Y.append([delta[2], 0., 0., 0.])
        Y.append([0., delta[3], delta[4], delta[5]])

        X.append([phi[0], phi[3], 0., 0., 0., 0., 0., 0., 0., 0.])
        X.append([phi[1], 0., phi[4], 0., 0., 0., 0., 0., 0., 0.])
        X.append([phi[2], 0., 0., phi[5], 0., 0., 0., 0., 0., 0.])
        X.append([0., 0., 0., 0., phi[6], phi[7], phi[8], phi[9], phi[10], phi[11]])
    return np.array(Y), np.array(X)


def lse_quadrotor(Delta_S, Phi):
    """theta_hat_ for the quadrotor (10 unknown scalars).

    Paper builds YY (N,4), XX (N,10); theta_hat = (pinv(XX)@YY).T is (4,10)
    (row = output channel, col = feature), and the 10 reported scalars come
    from specific (row, col) entries.
    """
    YY, XX = _quad_lse_matrices(Delta_S, Phi)
    M = (np.linalg.pinv(XX) @ YY).T
    return np.array([M[0, 0], M[0, 1], M[0, 2], M[0, 3],
                     M[1, 4], M[1, 7],
                     M[2, 5], M[2, 8],
                     M[3, 6], M[3, 9]])


# ---------------------------------------------------------------------------
# Set-membership estimation
# ---------------------------------------------------------------------------
def _halfspaces_pendulum(Delta_S, Phi, w_max, ground_truth):
    """Halfspaces of form: A theta <= b, assembled as [A | -b] <= 0 for
    HalfspaceIntersection (which needs A x + b <= 0)."""
    Ab = []
    for delta, phi in zip(Delta_S, Phi):
        Ab.append(np.r_[phi, -(w_max + delta)])
        Ab.append(np.r_[-phi, -(w_max - delta)])
    return _feasible_and_vertices(Ab, ground_truth)


def _halfspaces_quadrotor(Delta_S, Phi, w_max, fallback_point):
    """Halfspace constraints for the 10-D quadrotor uncertainty set.

    Mirrors set_membership_lin_prog_qdt.py: each of the 6 measured components
    produces two halfspaces |active_halfspaces(delta_c) - theta_pair . features| <= w_max.
    """
    # (feature index, theta index) pairs per component, matching the paper code:
    comp_feat_theta = {
        0: [(0, 0), (3, 1)],
        1: [(1, 0), (4, 2)],
        2: [(2, 0), (5, 3)],
        3: [(6, 4), (9, 5)],
        4: [(7, 6), (10, 7)],
        5: [(8, 8), (11, 9)],
    }
    Ab = []
    for delta, phi in zip(Delta_S, Phi):
        for c in range(6):
            row = np.zeros(10)
            for f, th in comp_feat_theta[c]:
                row[th] = phi[f]
            Ab.append(np.r_[row, -(w_max + delta[c])])
            Ab.append(np.r_[-row, -(w_max - delta[c])])
    return _feasible_and_vertices(Ab, fallback_point)


def _feasible_and_vertices(Ab, fallback_point):
    """Find an interior point inside the polytope A x <= b, then return the
    vertices of the halfspace intersection (HalfspaceIntersection vertices).

    Ab rows encode [A | -b] so that the polytope is A x <= b  <=>  Ab @ (x,1) <= 0.
    The interior point is the Chebyshev center: the point maximising the min
    slack, i.e.  max s  s.t. A x + s <= b, guaranteeing strict interiority
    (HalfspaceIntersection requires a strictly-interior point). If the polytope
    is degenerate/empty the paper falls back to the ground truth as the
    'feasible point'; we mirror that here.
    """
    Ab = np.array(Ab, dtype=float)
    A = Ab[:, :-1]
    b = -Ab[:, -1]
    n = A.shape[1]
    # Chebyshev center: max s s.t. A x + s <= b  (s scalar >= 0)
    c = np.concatenate([np.zeros(n), [-1.0]])
    A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
    res = linprog(c=c, A_ub=A_aug, b_ub=b, bounds=[(None, None)] * n + [(0, None)],
                  method="highs")
    if res.success:
        x0 = res.x[:n]
    else:
        x0 = np.asarray(fallback_point, dtype=float)
    hs = HalfspaceIntersection(Ab, x0)
    return hs.intersections


def run_sme(Delta_S, Phi, w_max, dim, ground_truth):
    """Compute the SME uncertainty-set vertices for the given data.

    Returns array of vertex points (rows). If dim == 2 the pendulum (2-D)
    polytope is computed; otherwise the quadrotor's 10-D polytope.
    """
    if dim == 2:
        return _halfspaces_pendulum(Delta_S, Phi, w_max, ground_truth)
    return _halfspaces_quadrotor(Delta_S, Phi, w_max, ground_truth)


def uncertainty_diameter(points):
    """Diameter of a point set = max pairwise Euclidean distance."""
    if len(points) < 2:
        return 0.0
    return float(np.max(distance_matrix(points, points)))
