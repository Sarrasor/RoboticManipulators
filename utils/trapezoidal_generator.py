"""
TrapezoidalGenerator class definition
"""
from math import ceil

import numpy as np


class TrapezoidalGenerator():
    """
    Trapezoidal Generator class generates trapezoidal speed profiles that
    consider velocity and acceleration limits
    """

    def __init__(self, dq_max, ddq_max, control_freq=0):
        """
        Initializes Trapezoidal Generator with velocity and acceleration limits

        Args:
            dq_max (float): Maximum velocity of joints
            ddq_max (float): Maximum acceleration of joints
            control_freq (int, optional): Frequency of a controller. If it is 0
                will not take the controller frequency into account
        """
        self._dq_max = dq_max
        self._ddq_max = ddq_max
        self._control_freq = control_freq

    def set_limits(self, dq_max, ddq_max):
        self._dq_max = dq_max
        self._ddq_max = ddq_max

    def set_frequency(self, control_freq):
        self._control_freq = control_freq

    def generate_coefficients(self, q_init, q_final):
        """
        Generates trapezoidal profile

        Args:
            q_init (float): Current position
            q_final (float): Desired position

        Returns:
            float, float:
                t_1 - acceleration stop time
                tau - deceleration start time

                  /----------\
                 / |        | \
                /  |        |  \
                  t_1       tau
        """
        self.set_positions(q_init, q_final)
        delta_q = abs(self._q_final - self._q_init)
        dq_max_prime = np.sqrt(delta_q * self._ddq_max)

        # Triangular profile
        if dq_max_prime < self._dq_max:
            t_1 = dq_max_prime / self._ddq_max
            tau = t_1
            self._dq_max_current = dq_max_prime
        # Trapezoidal profile
        else:
            t_1 = self._dq_max / self._ddq_max
            tau = delta_q / self._dq_max
            self._dq_max_current = self._dq_max

        self._ddq_max_current = self._ddq_max

        # Consider controller frequency
        if self._control_freq != 0:
            n = ceil(t_1 * self._control_freq)
            m = ceil(tau * self._control_freq)

            t_1 = n / self._control_freq
            tau = m / self._control_freq

        return t_1, tau

    def set_current_limits(self, dq_max_current, ddq_max_current):
        """
        Updates current velocity and acceleration limits

        Args:
            dq_max_current (float): Current maximum velocity. Can be negative
                if the initial position is larger than the final position
            ddq_max_current (float): Current maximum acceleration. Can
                be negative if the initial position is larger than the
                final position
        """
        self._dq_max_current = dq_max_current
        self._ddq_max_current = ddq_max_current

    def set_positions(self, q_init, q_final):
        """
        Updates current and desired positions

        Args:
            q_init (float): Current position
            q_final (float): Desired position
        """
        self._q_init = q_init
        self._q_final = q_final

    @property
    def dq_max_current(self):
        return self._dq_max_current

    def get_t(self, t_1, tau, n):
        """
        Generates discretized time profile with n points from t_1 and tau

        Args:
            t_1 (float): Acceleration end time
            tau (float): Deceleration start time
            n (int): Number of points

        Returns:
            np.ndarray: Time profile with n points
        """
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)
        return np.hstack([t_acc, t_no_acc, t_dec])

    def get_q(self, t_1, tau, n):
        """
        Samples n points from joint position profile of t_1 and tau

        Args:
            t_1 (float): Acceleration end time
            tau (float): Deceleration start time
            n (int): Number of points

        Returns:
            np.ndarray: N samples of joint positions for t_1 and tau

        Raises:
            ValueError: Raises this error if the number of requested points is
                too small for current velocity and acceleration limits
        """
        t_acc, t_no_acc, _ = self._get_times(t_1, tau, n)
        t_no_acc = t_no_acc - t_acc[-1]

        if len(t_acc) == 0:
            raise ValueError("Not enough points. Please, increase n")

        q_acc = self._q_init + 0.5 * self._ddq_max_current * t_acc**2
        q_no_acc = q_acc[-1] + self._dq_max_current * t_no_acc

        # Check if profile is trapezoidal or triangular
        if len(q_no_acc) != 0:
            start_q = q_no_acc[-1]
        else:
            start_q = q_acc[-1]

        q_dec = start_q + self._dq_max_current * \
            t_acc - 0.5 * self._ddq_max_current * t_acc**2

        return np.hstack([q_acc, q_no_acc, q_dec])

    def get_dq(self, t_1, tau, n):
        """
        Samples n points from velocity profile of t_1 and tau

        Args:
            t_1 (float): Acceleration end time
            tau (float): Deceleration start time
            n (int): Number of points

        Returns:
            np.ndarray: N samples of joint velocities for t_1 and tau

        Raises:
            ValueError: Raises this error if the number of requested points is
                too small for current velocity and acceleration limits
        """
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)

        if len(t_acc) == 0:
            raise ValueError("Not enough points. Please, increase n")

        q_acc = self._ddq_max_current * t_acc
        q_no_acc = np.ones(len(t_no_acc)) * self._dq_max_current
        q_dec = q_acc[-1] - q_acc

        return np.hstack([q_acc, q_no_acc, q_dec])

    def get_ddq(self, t_1, tau, n):
        """
        Samples n points from acceleration profile of t_1 and tau

        Args:
            t_1 (float): Acceleration end time
            tau (float): Deceleration start time
            n (int): Number of points

        Returns:
            np.ndarray: N samples of joint accelerations for t_1 and tau
        """
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)

        q_acc = self._ddq_max_current * np.ones(len(t_acc))
        q_no_acc = np.zeros(len(t_no_acc))
        q_dec = -self._ddq_max_current * np.ones(len(t_dec))

        return np.hstack([q_acc, q_no_acc, q_dec])

    def _get_times(self, t_1, tau, n):
        """
        Samples n points from 0 to (tau + t_1) time interval

        Args:
            t_1 (float): Acceleration end time
            tau (float): Deceleration start time
            n (int): Number of points

        Returns:
            np.ndarray, np.ndarray, np.ndarray: Times of acceleration,
                Times of no acceleration, Times of deceleration
        """
        step = n / (tau + t_1)

        t_acc = np.linspace(0, t_1, int(step * t_1))
        t_no_acc = np.linspace(t_1, tau, int(step * (tau - t_1)))
        t_dec = np.linspace(tau, tau + t_1, int(step * t_1))

        return t_acc, t_no_acc, t_dec
