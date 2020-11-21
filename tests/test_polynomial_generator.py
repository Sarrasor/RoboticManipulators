import unittest

import numpy as np
from utils.polynomial_generator import PolynomialGenerator


class TestPolynomialGenerator(unittest.TestCase):
    epsilon = 0.001
    pg = PolynomialGenerator()

    def test_generator(self):

        q_init = np.array([0, 0, 0])
        q_final = np.array([1, 0, 0])
        t = 1

        real_coefs = np.array([0.0, 0.0, 0.0, 10.0, -15.0, 6.0])

        coefs = self.pg.generate_coefficients(q_init, q_final, t)

        self.assertTrue(np.all((real_coefs - coefs) <= self.epsilon))

    def test_derivative(self):
        q_init = np.array([0, 0, 0])
        q_final = np.array([1, 0, 0])
        from_t = 0
        to_t = 1
        n = 100

        coefs = self.pg.generate_coefficients(q_init,
                                              q_final,
                                              to_t,
                                              t_0=from_t)
        poly = self.pg.polynomial_from_coefs(coefs, from_t, to_t, n)
        dpoly = self.pg.dpolynomial_from_coefs(0, coefs, from_t, to_t, n)

        self.assertTrue(np.all((poly - dpoly) <= self.epsilon))
