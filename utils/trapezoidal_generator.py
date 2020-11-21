from math import ceil
import numpy as np


class TrapezoidalGenerator():
    def __init__(self, dq_max, ddq_max, control_freq=0):
        self._dq_max = abs(dq_max)
        self._ddq_max = abs(ddq_max)
        self._control_freq = control_freq

    def generate_coefficients(self, q_init, q_final):
        self.set_positions(q_init, q_final)
        delta_q = abs(self._q_final - self._q_init)

        dq_max_prime = np.sqrt(delta_q * self._ddq_max)

        if dq_max_prime < self._dq_max:
            t_1 = dq_max_prime / self._ddq_max
            tau = t_1
            self._dq_max_current = dq_max_prime
        else:
            t_1 = self._dq_max / self._ddq_max
            tau = delta_q / self._dq_max
            self._dq_max_current = self._dq_max

        self._ddq_max_current = self._ddq_max

        if self._control_freq != 0:
            n = ceil(t_1 * self._control_freq) + 1
            m = ceil(tau * self._control_freq) + 1

            t_1 = n / self._control_freq
            tau = m / self._control_freq

        return t_1, tau

    def set_current_limits(self, dq_max_current, ddq_max_current):
        self._dq_max_current = dq_max_current
        self._ddq_max_current = ddq_max_current

    def set_positions(self, q_init, q_final):
        self._q_init = q_init
        self._q_final = q_final

    @property
    def dq_max_current(self):
        return self._dq_max_current

    def get_t(self, t_1, tau, n):
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)
        return np.hstack([t_acc, t_no_acc, t_dec])

    def get_q(self, t_1, tau, n):
        t_acc, t_no_acc, _ = self._get_times(t_1, tau, n)
        t_no_acc = t_no_acc - t_acc[-1]

        q_acc = self._q_init + 0.5 * self._ddq_max_current * t_acc**2
        q_no_acc = q_acc[-1] + self._dq_max_current * t_no_acc
        if len(q_no_acc) != 0:
            start_q = q_no_acc[-1]
        else:
            start_q = q_acc[-1]
        q_dec = start_q + self._dq_max_current * \
            t_acc - 0.5 * self._ddq_max_current * t_acc**2

        return np.hstack([q_acc, q_no_acc, q_dec])

    def get_dq(self, t_1, tau, n):
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)

        q_acc = self._ddq_max_current * t_acc
        q_no_acc = np.ones(len(t_no_acc)) * self._dq_max_current
        q_dec = q_acc[-1] - q_acc

        return np.hstack([q_acc, q_no_acc, q_dec])

    def get_ddq(self, t_1, tau, n):
        t_acc, t_no_acc, t_dec = self._get_times(t_1, tau, n)

        q_acc = self._ddq_max_current * np.ones(len(t_acc))
        q_no_acc = np.zeros(len(t_no_acc))
        q_dec = -self._ddq_max_current * np.ones(len(t_dec))

        return np.hstack([q_acc, q_no_acc, q_dec])

    def _get_times(self, t_1, tau, n):
        step = n / (tau + t_1)

        t_acc = np.linspace(0, t_1, int(step * t_1))
        t_no_acc = np.linspace(t_1, tau, int(step * (tau - t_1)))
        t_dec = np.linspace(tau, tau + t_1, int(step * t_1))

        return t_acc, t_no_acc, t_dec
