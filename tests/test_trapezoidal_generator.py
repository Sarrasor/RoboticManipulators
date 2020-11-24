import unittest

import numpy as np
from utils.trapezoidal_generator import TrapezoidalGenerator


class TestTrapezoidalGenerator(unittest.TestCase):
    epsilon = 0.001
    tg = TrapezoidalGenerator(dq_max=1, ddq_max=10)

    def test_generator(self):
        self.tg.set_limits(2, 2)
        q_init = 0
        q_final = 5

        real_coefs = np.array([1.0, 2.5])

        t_1, tau = self.tg.generate_coefficients(q_init, q_final)
        coefs = np.array([t_1, tau])
        self.assertTrue(np.all(np.abs((real_coefs - coefs)) <= self.epsilon))

    def test_generator_frequency(self):
        self.tg.set_limits(2, 2)
        self.tg.set_frequency(5)
        q_init = 0
        q_final = 5

        real_coefs = np.array([1.0, 2.6])

        t_1, tau = self.tg.generate_coefficients(q_init, q_final)
        coefs = np.array([t_1, tau])
        self.assertTrue(np.all(np.abs((real_coefs - coefs)) <= self.epsilon))
