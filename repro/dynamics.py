"""
Reproduction of the dynamics in:

    Identification of Analytic Nonlinear Dynamical Systems with
    Non-asymptotic Guarantees   (NeurIPS 2024, arXiv:2411.0656)
    Negin Musavi, Ziyao Guo, Geir Dullerud, Yingying Li

Implements the pendulum (Example 1) and quadrotor (Example 2) discrete-time
dynamics exactly as in the paper's official source code
(https://github.com/NeginMusavi/real-analytic-nonlinear-sys-id).

Both systems are linearly parameterised nonlinear systems of the form
    s_{t+1} - s_t - ex_t = theta* phi(z_t) + w_t,   z_t = (x_t, u_t)
where s_t is the "derivative" state being regressed, ex_t a known affine offset
(gravity), phi(z_t) the known analytic feature vector, theta* the unknown
parameters, and w_t the i.i.d. exploration/perturbation disturbance.
"""

import math
import numpy as np
from scipy.stats import truncnorm

G = 9.81          # gravitational acceleration (m/s^2)
DT = 0.01         # discretization time step (s)


def _noise1(distr, time_hor, seed, mean, std, support):
    """i.i.d. scalar noise, support [-support, support]."""
    rng = np.random.RandomState(seed)
    if distr == "uniform":
        return support * rng.uniform(-1.0, 1.0, time_hor)
    if distr == "trunc_guass":
        if support is None:
            raise ValueError("trunc_guass needs a support bound")
        return truncnorm(-support, support, loc=mean, scale=std).rvs(size=time_hor,
                                                                     random_state=rng)
    raise ValueError("unknown distribution: " + distr)


class Pendulum:
    """Single pendulum. State x = (alpha, omega). theta* = (1/l, 1/(m l^2))."""

    def __init__(self, k=2.0, m=0.1, l=0.5):
        self.m, self.l, self.k = m, l, k

    @property
    def theta_star(self):
        return np.array([1.0 / self.l, 1.0 / (self.m * self.l * self.l)])

    def trajectory(self, time_hor, distr, seed_u, seed_w,
                   param_input, param_dist, mult_u=(1.0,), mult_w=(1.0, 1.0)):
        """Generate one trajectory. Returns (Delta_S, Phi_S_U) lists of length T."""
        # control-noise support: paper uses [-u_max, u_max]
        u_support = param_input[2] if distr == "trunc_guass" else param_input[1]
        # disturbance support
        w_support = param_dist[2] if distr == "trunc_guass" else param_dist[1]

        U = _noise1(distr, time_hor, seed_u, param_input[0], param_input[1], u_support) * mult_u[0]
        W1 = _noise1(distr, time_hor, seed_w, param_dist[0], param_dist[1], w_support) * mult_w[0]
        W2 = _noise1(distr, time_hor, seed_w, param_dist[0], param_dist[1], w_support) * mult_w[1]

        alpha, omega = 0.0, 0.0
        Delta_S, Phi = [], []

        for t in range(time_hor):
            s_ = omega
            u1 = U[t]
            w1 = W1[t]
            w2 = W2[t]
            u = -self.k * omega + u1

            alpha_dot = omega + w1
            omega_dot = -G * math.sin(alpha) / self.l + u / (self.m * self.l * self.l) + w2

            phi_s_u = np.array([-G * math.sin(alpha), u])

            alpha = alpha + DT * alpha_dot
            omega = omega + DT * omega_dot

            s = omega
            Delta_S.append(s - s_)
            Phi.append(DT * phi_s_u)

        return Delta_S, Phi


class Quadrotor:
    """Quadrotor (Alaimo et al., 2013). x=(p,v,q,omega). theta* is a 6x12 matrix."""

    def __init__(self):
        # inertia (kg m^2), aerodynamic drag (kg/s), mass (kg)
        self.Ixx, self.Iyy, self.Izz = 4.856e-3, 4.856e-3, 8.801e-3
        self.Ax, self.Ay, self.Az = 0.25, 0.25, 0.25
        self.m = 0.468

    @property
    def theta_star_vec(self):
        """"The vector of 10 unknown scalars used by the paper's LSE/SME."""
        Ixx, Iyy, Izz = self.Ixx, self.Iyy, self.Izz
        m, Ax, Ay, Az = self.m, self.Ax, self.Ay, self.Az
        return np.array([1 / m,
                         -Ax / m, -Ay / m, -Az / m,
                         (Iyy - Izz) / Ixx, 1 / Ixx,
                         (Izz - Ixx) / Iyy, 1 / Iyy,
                         (Ixx - Izz) / Izz, 1 / Izz])

    @property
    def theta_star_matrix(self):
        Ixx, Iyy, Izz = self.Ixx, self.Iyy, self.Izz
        m, Ax, Ay, Az = self.m, self.Ax, self.Ay, self.Az
        return np.array([[1 / m, 0., 0., -Ax / m, 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 1 / m, 0., 0., -Ay / m, 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1 / m, 0., 0., -Az / m, 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., (Iyy - Izz) / Ixx, 0., 0., 1 / Ixx, 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., (Izz - Ixx) / Iyy, 0., 0., 1 / Iyy, 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., (Iyy - Ixx) / Izz, 0., 0., 1 / Izz]])

    def _euler_to_quaternion(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        return (cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy)

    def trajectory(self, time_hor, distr, seed_u, seed_w,
                   param_input, param_dist, mult_u=(1., 0.2, 0.2, 0.2)):
        """Generate one trajectory. Returns (Delta_S, Phi_S_U) lists of length T."""
        u_support = param_input[2] if distr == "trunc_guass" else param_input[1]
        w_support = param_dist[2] if distr == "trunc_guass" else param_dist[1]

        U1 = _noise1(distr, time_hor, seed_u, param_input[0], param_input[1], u_support)
        U2 = _noise1(distr, time_hor, seed_u, param_input[0], param_input[1], u_support)
        U3 = _noise1(distr, time_hor, seed_u, param_input[0], param_input[1], u_support)
        U4 = _noise1(distr, time_hor, seed_u, param_input[0], param_input[1], u_support)
        u1 = mult_u[0] * U1; u2 = mult_u[1] * U2; u3 = mult_u[2] * U3; u4 = mult_u[3] * U4

        W = [_noise1(distr, time_hor, seed_w, param_dist[0], param_dist[1], w_support)
             for _ in range(6)]

        # initial states (identity attitude)
        p = np.array([0.0, 0.0, 1.0])
        v = np.array([0.0, 0.0, 0.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([10.0, 10.0, 10.0])

        J = np.diag([self.Ixx, self.Iyy, self.Izz])
        omega_ = omega.copy()

        # controller gains / reference
        kp_z, kd_z = 0.75, 1.25
        kp_phi = kp_theta = kp_psi = 0.03
        kd_phi = kd_theta = kd_psi = 0.00875
        pz_d, vz_d = 5.0, 0.0
        q0_d, q1_d, q2_d, q3_d = self._euler_to_quaternion(0, 0, 0)
        gvec = np.array([0.0, 0.0, G, 0.0, 0.0, 0.0])

        Delta_S, Phi = [], []
        t = 0
        while t < time_hor:
            q0, q1, q2, q3 = q
            w1, w2, w3, w4, w5, w6 = (W[i][t] for i in range(6))

            # control with exploration noise
            pi_z = kp_z * (pz_d - p[2]) + kd_z * (vz_d - v[2])
            f_c = np.array([0.0, 0.0, (5.0 + pi_z + u1[t])])

            qe1 = -q0_d*q1 - q3_d*q2 + q2_d*q3 + q1_d*q0
            qe2 =  q3_d*q1 - q0_d*q2 - q1_d*q3 + q2_d*q0
            qe3 = -q2_d*q1 + q1_d*q2 - q0_d*q3 + q3_d*q0
            qe4 =  q1_d*q1 + q2_d*q2 + q3_d*q3 + q0_d*q0
            pi_phi = -kd_phi*omega[0] + kp_phi*qe1*qe4
            pi_theta = -kd_theta*omega[1] + 2*kp_theta*qe2*qe4
            pi_psi = -kd_psi*omega[2] + 2*kp_psi*qe3*qe4
            tau_c = np.array([pi_phi + u2[t], pi_theta + u3[t], pi_psi + u4[t]])

            Q = np.array([[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q0*q2+q1*q3)],
                          [2*(q1*q2+q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
                          [2*(q1*q3-q0*q2), 2*(q0*q1+q2*q3), q0*q0-q1*q1-q2*q2+q3*q3]])
            Om = np.array([[0,-omega[0],-omega[1],-omega[2]],
                           [omega[0],0,omega[2],-omega[1]],
                           [omega[1],-omega[2],0,omega[0]],
                           [omega[2],omega[1],-omega[0],0]])

            Qfc = Q @ f_c
            phi_s_u = np.array([Qfc[0], Qfc[1], Qfc[2], v[0], v[1], v[2],
                                omega[1]*omega[2], omega[0]*omega[2], omega[0]*omega[1],
                                tau_c[0], tau_c[1], tau_c[2]])

            p_dot = v
            q_dot = Om @ q / 2
            s_dot = -gvec + self.theta_star_matrix @ phi_s_u + np.array([w1, w2, w3, w4, w5, w6])

            p_new = p + DT * p_dot
            q_new = (q + DT * q_dot) / np.sum(q*q)
            s_new = np.concatenate([v, omega]) + DT * s_dot

            v_new = s_new[:3]
            omega_new = s_new[3:]

            Delta_S.append((s_new - np.concatenate([v, omega])) + DT * gvec)
            Phi.append(DT * phi_s_u)

            p, q = p_new, q_new
            v, omega = v_new, omega_new
            t += 1

        return Delta_S, Phi
