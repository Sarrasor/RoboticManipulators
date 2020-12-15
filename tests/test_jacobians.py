import unittest

import numpy as np
import sympy as sp

from utils.jacobians import JacobianCalculator


class TestJacobianCalculatorModule(unittest.TestCase):
    def testEqualityOfMethods(self):
        jc = JacobianCalculator("RzTx", [1, 0], simplify=False)
        J_num = jc.calculate_numeric()
        J_skew = jc.calculate_scew()
        Res = np.array(sp.simplify(J_skew - J_num), dtype=np.float)

        self.assertTrue(np.allclose(Res, np.zeros((6, 1))))

        jc = JacobianCalculator("RzTxRzTx", [1, 0, 1, 0], simplify=False)
        J_num = jc.calculate_numeric()
        J_skew = jc.calculate_scew()
        Res = np.array(sp.simplify(J_skew - J_num), dtype=np.float)

        self.assertTrue(np.allclose(Res, np.zeros((6, 2))))

        jc = JacobianCalculator("RzTxRzTxTyRz", [1, 1, 1, 1, 0, 1])
        J_num = jc.calculate_numeric()
        J_skew = jc.calculate_scew()
        Res = np.array(sp.simplify(J_skew - J_num), dtype=np.float)

        self.assertTrue(np.allclose(Res, np.zeros((6, 5))))

    def testWithTransforms(self):
        T_base = sp.Matrix([
            [1, 0, 0, 3],
            [0, 1, 0, 6],
            [0, 0, 1, -9],
            [0, 0, 0, 1]
        ])

        T_tool = sp.Matrix([
            [0, 1, 0, 2],
            [1, 0, 0, -4],
            [0, 0, 1, 6],
            [0, 0, 0, 1]
        ])

        jc = JacobianCalculator("RzTxRzTx",
                                [1, 0, 1, 0],
                                T_base=T_base,
                                T_tool=T_tool,
                                simplify=False)
        J_num = jc.calculate_numeric()
        J_skew = jc.calculate_scew()

        Res = np.array(sp.simplify(J_skew - J_num), dtype=np.float)

        self.assertTrue(np.allclose(Res, np.zeros((6, 2))))

    def testNumericValues(self):
        jc = JacobianCalculator("RzTxRzTx",
                                [1, 0, 1, 0],
                                simplify=False)

        values = (0.0, 1.0, 0.0, 2.0)
        J = jc.calculate_numerically(values)
        J = np.array(J, dtype=np.float)

        RealJ = np.array([[0.0, 0.0],
                          [3.0, 2.0],
                          [0.0, 0.0],
                          [0.0, 0.0],
                          [0.0, 0.0],
                          [1.0, 1.0]])

        self.assertTrue(np.allclose(J, RealJ))
