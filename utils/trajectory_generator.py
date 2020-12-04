"""
TrajectoryGenerator class definition
"""
import numpy as np
import sympy as sp

from utils.polynomial_generator import PolynomialGenerator
from utils.plot_utils import TrajectoriesPlotter
from utils.trapezoidal_generator import TrapezoidalGenerator


class TrajectoryGenerator():
    """
    Trajectory Generator is able to generate polynomial, p2p joint space
    trajectory, and linear cartesian space trajectory for robotic manipulators
    """

    def __init__(self, dq_max, ddq_max, dx_max, ddx_max, control_freq=0):
        self._pg = PolynomialGenerator()
        self._tg = TrapezoidalGenerator(dq_max=dq_max, ddq_max=ddq_max)
        self._dq_max = dq_max
        self._ddq_max = ddq_max
        self._dx_max = dx_max
        self._ddx_max = ddx_max
        self._control_freq = control_freq

    def generate_joint_poly_trajectory(self,
                                       qs_0,
                                       qs_f,
                                       t_f,
                                       t_0=0.0,
                                       n=100,
                                       plot=True):
        """
        Generates quintic polynomial trajectory profile given desired
        initial and final conditions

        Args:
            qs_0 (list of lists of 3 floats): List of initial positions,
                velocities, accelerations of all joints at time t_0
            qs_f (list of lists of 3 floats): List of desired positions,
                velocities, accelerations of all joints at time t_f
            t_f (float): End time
            t_0 (float, optional): Start time
            n (int, optional): Number of points to sample
            plot (bool, optional): Flag to plot profiles

        Returns:
            np.ndarray: Array of n joint positions for each joint
        """

        # Generate polynomial coefficients
        coefs = []
        for q_0, q_f in zip(qs_0, qs_f):
            coefs.append(self._pg.generate_coefficients(q_0, q_f, t_f, t_0))

        # Generate profiles from coefficients
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
        """
        Generates p2p trajectory in joint space

        Args:
            qs_0 (list of float): List of initial positions of all joints
            qs_f (list of float): List of desired positions of all joints
            n (int, optional): Number of points to sample
            plot (bool, optional): Flag to plot profiles

        Returns:
            np.ndarray: Array of n joint positions for each joint
        """
        self._tg.set_frequency(self._control_freq)
        self._tg.set_limits(self._dq_max, self._ddq_max)

        qs, dqs, ddqs, ts = self._generate_equalized_profiles(qs_0, qs_f, n)

        if plot:
            TrajectoriesPlotter.plot_joint(ts, qs, dqs, ddqs)

        return np.array(qs).T

    def generate_lin_trajectory(self, p_0, p_f, n=100, plot=False):
        """
        Generates linear trajectory between b_0 and p_f

        Args:
            p_0 (list of 3 floats): Initial 3D point
            p_f (list of 3 floats): Desired 3D point
            n (int, optional): Number of points to sample
            plot (bool, optional): Flag to plot profiles

        Returns:
            np.ndarray, np.ndarray: Array of n cartesian positions for each
                3D axis, Array of n cartesian velocities for each 3D axis
        """
        self._tg.set_frequency(self._control_freq)
        self._tg.set_limits(self._dx_max, self._ddx_max)

        xs, dxs, ddxs, ts = self._generate_equalized_profiles(p_0, p_f, n)

        if plot:
            TrajectoriesPlotter.plot_cartesian(ts, xs, dxs, ddxs)

        return np.array(xs).T, np.array(dxs).T, ts

    def _generate_equalized_profiles(self, qs_0, qs_f, n):
        """
        Generates equalized trapezoidal profiles for several joints

        Args:
            qs_0 (list of float): List of initial positions of all joints
            qs_f (list of float): List of desired positions of all joints
            n (int): Number of points to sample

        Returns:
            np.ndarray, np.ndarray, np.ndarray, float, float:
                qs - Joint positions
                dqs - Joint velocities
                ddqs - Joint accelerations
                ts - Corresponding times
        """

        # Pregenerate coefficients
        ts_1, taus = [], []
        for q_0, q_f in zip(qs_0, qs_f):
            t_1, tau = self._tg.generate_coefficients(q_0, q_f)

            ts_1.append(t_1)
            taus.append(tau)

        # Equalize profiles
        t_1_max = max(ts_1)
        tau_max = max(taus)
        ts = self._tg.get_t(t_1_max, tau_max, n)
        qs, dqs, ddqs = [], [], []
        for q_0, q_f in zip(qs_0, qs_f):
            delta_q = q_f - q_0
            dq_cur = delta_q / tau_max
            ddq_cur = delta_q / (tau_max * t_1_max)

            self._tg.set_current_limits(dq_cur, ddq_cur)
            self._tg.set_positions(q_0, q_f)
            qs.append(self._tg.get_q(t_1_max, tau_max, n))
            dqs.append(self._tg.get_dq(t_1_max, tau_max, n))
            ddqs.append(self._tg.get_ddq(t_1_max, tau_max, n))

        return qs, dqs, ddqs, ts

    def get_dq_from_dx(self, qs, dxs, J_inv, ts=None):
        """
        Generates joint velocities from cartesian velocities with help of
        Jacobian

        Args:
            qs (np.ndarray): Joint positions array
            dxs (np.ndarray): Cartesian velocities array
            J_inv (sp.Matrix): Symbolic Jacobian sympy Matrix
            ts (None, optional): Provide corresponding times if plot is needed

        Returns:
            np.ndarray: Joint velocities
        """
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
        """
        Produces linear interpolation between two 3D points

        Args:
            p_start (list of 3 floats): Description
            p_finish (list of 3 floats): Description
            n (int, optional): Number of points to sample

        Returns:
            np.array: Interpolated points
        """
        v = np.array([p_finish]) - np.array([p_start])
        t = np.array([np.linspace(0, 1, n)]).T
        return p_start + t.dot(v)
