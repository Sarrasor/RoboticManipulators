import numpy as np
import sympy as sp

from utils.polynomial_generator import PolynomialGenerator
from utils.plot_utils import TrajectoriesPlotter
from utils.trapezoidal_generator import TrapezoidalGenerator


class TrajectoryGenerator():
    def __init__(self, dq_max, ddq_max, dx_max, ddx_max, control_freq=0):
        self._pg = PolynomialGenerator()
        self._dq_max = dq_max
        self._ddq_max = ddq_max
        self._dx_max = dx_max
        self._ddx_max = ddx_max
        self._control_freq = control_freq

    def generate_joint_poly_trajectory(self,
                                       qs_0,
                                       qs_f,
                                       t_f,
                                       t_0=0,
                                       n=100,
                                       plot=True):
        coefs = []
        for q_0, q_f in zip(qs_0, qs_f):
            coefs.append(self._pg.generate_coefficients(q_0, q_f, t_f, t_0))

        qs, dqs, ddqs = [], [], []

        for c in coefs:
            qs.append(self._pg.polynomial_from_coefs(c, t_0, t_f, n))
            if plot:
                dqs.append(self._pg.dpolynomial_from_coefs(1, c, t_0, t_f, n))
                ddqs.append(self._pg.dpolynomial_from_coefs(2, c, t_0, t_f, n))

        if plot:
            ts = np.linspace(t_0, t_f, n)
            TrajectoriesPlotter.plot_joint(ts, qs, dqs, ddqs)

        return np.array(qs).T

    def generate_p2p_trajectory(self, qs_0, qs_f, n=100, plot=False):
        trapez_gen = TrapezoidalGenerator(self._dq_max,
                                          self._ddq_max,
                                          self._control_freq)

        ts_1, taus = [], []
        for q_0, q_f in zip(qs_0, qs_f):
            t_1, tau = trapez_gen.generate_coefficients(q_0, q_f)

            ts_1.append(t_1)
            taus.append(tau)

        t_1_max = max(ts_1)
        tau_max = max(taus)

        qs, dqs, ddqs = [], [], []
        for q_0, q_f in zip(qs_0, qs_f):
            delta_q = q_f - q_0
            dq_cur = delta_q / tau_max
            ddq_cur = delta_q / (tau_max * t_1_max)

            trapez_gen.set_current_limits(dq_cur, ddq_cur)
            trapez_gen.set_positions(q_0, q_f)
            qs.append(trapez_gen.get_q(t_1_max, tau_max, n))

            if plot:
                dqs.append(trapez_gen.get_dq(t_1_max, tau_max, n))
                ddqs.append(trapez_gen.get_ddq(t_1_max, tau_max, n))

        if plot:
            ts = trapez_gen.get_t(t_1_max, tau_max, n)
            TrajectoriesPlotter.plot_joint(ts, qs, dqs, ddqs)

        return np.array(qs).T

    def generate_lin_trajectory(self, p_0, p_f, n=100, plot=True):
        trapez_gen = TrapezoidalGenerator(self._dx_max,
                                          self._ddx_max,
                                          self._control_freq)

        ts_1, taus = [], []
        for p_0_axis, p_f_axis in zip(p_0, p_f):
            t_1, tau = trapez_gen.generate_coefficients(p_0_axis, p_f_axis)

            ts_1.append(t_1)
            taus.append(tau)

        t_1_max = max(ts_1)
        tau_max = max(taus)

        xs, dxs, ddxs = [], [], []
        for p_0_axis, p_f_axis in zip(p_0, p_f):
            delta_x = p_f_axis - p_0_axis
            dx_cur = delta_x / tau_max
            ddx_cur = delta_x / (tau_max * t_1_max)

            trapez_gen.set_current_limits(dx_cur, ddx_cur)
            trapez_gen.set_positions(p_0_axis, p_f_axis)
            xs.append(trapez_gen.get_q(t_1_max, tau_max, n))
            dxs.append(trapez_gen.get_dq(t_1_max, tau_max, n))

            if plot:
                ddxs.append(trapez_gen.get_ddq(t_1_max, tau_max, n))

        if plot:
            ts = trapez_gen.get_t(t_1_max, tau_max, n)
            TrajectoriesPlotter.plot_cartesian(ts, xs, dxs, ddxs)

        return np.array(xs).T, np.array(dxs).T, ts

    def get_dq_from_dx(self, qs, dxs, J_inv, ts=None):
        dqs = []
        for dx, q in zip(dxs, qs.T):
            qs_dict = {}
            for i in range(len(q)):
                qs_dict[sp.symbols(f"q_{i}")] = q[i]

            dqs.append(np.array(J_inv.evalf(subs=qs_dict) * sp.Matrix(dx),
                                np.float))
        dqs = np.array(dqs)[:, :, 0].T

        if ts is not None:
            TrajectoriesPlotter.plot_joint_no_acc(ts, qs, dqs)

        return dqs

    def interpolate(self, p_start, p_finish, n):
        v = np.array([p_finish]) - np.array([p_start])
        t = np.array([np.linspace(0, 1, n)]).T
        return p_start + t.dot(v)
