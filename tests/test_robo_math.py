import unittest
import numpy as np

from utils.robo_math import Point3D
from utils.robo_math import Transformation, SymbolicTransformation


class TestRoboMathModule(unittest.TestCase):
    def test_point3d(self):
        point = Point3D(1.0, -2.0, 3.33)
        self.assertAlmostEqual(point.x, 1.0)
        self.assertAlmostEqual(point.y, -2.0)
        self.assertAlmostEqual(point.z, 3.33)

    def test_point3d_homogeneous(self):
        point = Point3D(2, 3, 4)
        point_hom = point.get_homogeneous()

        self.assertAlmostEqual(point_hom[0], 2.0)
        self.assertAlmostEqual(point_hom[1], 3.0)
        self.assertAlmostEqual(point_hom[2], 4.0)
        self.assertAlmostEqual(point_hom[3], 1.0)

    def test_trainsformation_Rx_degrees(self):
        Rx_0 = Transformation.get_Rx_from_degrees(0)
        Rx_0 = np.array(Rx_0, dtype=np.float)
        self.assertTrue(np.allclose(Rx_0, np.eye(4)))

        Rx_90_real = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        Rx_90 = Transformation.get_Rx_from_degrees(90)
        Rx_90 = np.array(Rx_90, dtype=np.float)

        self.assertTrue(np.allclose(Rx_90, Rx_90_real))

    def test_transformation_Ry_degrees(self):
        Ry_0 = Transformation.get_Ry_from_degrees(0)
        Ry_0 = np.array(Ry_0, dtype=np.float)
        self.assertTrue(np.allclose(Ry_0, np.eye(4)))

        Ry_90_real = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        Ry_90 = Transformation.get_Ry_from_degrees(90)
        Ry_90 = np.array(Ry_90, dtype=np.float)

        self.assertTrue(np.allclose(Ry_90, Ry_90_real))

    def test_transformation_Rz_degrees(self):
        Rz_0 = Transformation.get_Rz_from_degrees(0)
        Rz_0 = np.array(Rz_0, dtype=np.float)
        self.assertTrue(np.allclose(Rz_0, np.eye(4)))

        Rz_90_real = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        Rz_90 = Transformation.get_Rz_from_degrees(90)
        Rz_90 = np.array(Rz_90, dtype=np.float)

        self.assertTrue(np.allclose(Rz_90, Rz_90_real))

    def test_transformation_Tx(self):
        Tx_0 = Transformation.get_Tx(0)
        Tx_0 = np.array(Tx_0, dtype=np.float)

        self.assertTrue(np.allclose(Tx_0, np.eye(4)))

        Tx_10 = Transformation.get_Tx(10)
        Tx_10 = np.array(Tx_10, dtype=np.float)

        Tx_10_real = np.eye(4)
        Tx_10_real[0, 3] = 10

        self.assertTrue(np.allclose(Tx_10, Tx_10_real))

    def test_transformation_Ty(self):
        Ty_0 = Transformation.get_Ty(0)
        Ty_0 = np.array(Ty_0, dtype=np.float)

        self.assertTrue(np.allclose(Ty_0, np.eye(4)))

        Ty_10 = Transformation.get_Ty(10)
        Ty_10 = np.array(Ty_10, dtype=np.float)

        Ty_10_real = np.eye(4)
        Ty_10_real[1, 3] = 10

        self.assertTrue(np.allclose(Ty_10, Ty_10_real))

    def test_transformation_Tz(self):
        Tz_0 = Transformation.get_Tz(0)
        Tz_0 = np.array(Tz_0, dtype=np.float)

        self.assertTrue(np.allclose(Tz_0, np.eye(4)))

        Tz_10 = Transformation.get_Tz(10)
        Tz_10 = np.array(Tz_10, dtype=np.float)

        Tz_10_real = np.eye(4)
        Tz_10_real[2, 3] = 10

        self.assertTrue(np.allclose(Tz_10, Tz_10_real))

    def test_sym_transformation_Tx(self):
        Tx_sym = SymbolicTransformation.get_Tx()
        Txx_sym = Tx_sym * Tx_sym
        x_str = str(Txx_sym[0, 3])
        self.assertTrue(x_str == "2*x" or x_str == "x+x")

    def test_sym_transformation_Ty(self):
        Ty_sym = SymbolicTransformation.get_Ty('y_var')
        self.assertEqual(str(Ty_sym[1, 3]), 'y_var')

    def test_sym_transformation_Tz(self):
        Tz_sym = SymbolicTransformation.get_Tz()
        self.assertEqual(str(Tz_sym[2, 3]), 'z')

    def test_sym_transformation_Rx(self):
        Rx_sym = SymbolicTransformation.get_Rx()
        self.assertAlmostEqual(Rx_sym[0, 0], 1.0)
        self.assertEqual(str(Rx_sym[1, 1]), "cos(q)")
        self.assertEqual(str(Rx_sym[2, 2]), "cos(q)")
        self.assertEqual(str(Rx_sym[1, 2]), "-sin(q)")
        self.assertEqual(str(Rx_sym[2, 1]), "sin(q)")

    def test_sym_transformation_Ry(self):
        Ry_sym = SymbolicTransformation.get_Ry()
        self.assertAlmostEqual(Ry_sym[1, 1], 1.0)
        self.assertEqual(str(Ry_sym[0, 0]), "cos(q)")
        self.assertEqual(str(Ry_sym[2, 2]), "cos(q)")
        self.assertEqual(str(Ry_sym[2, 0]), "-sin(q)")
        self.assertEqual(str(Ry_sym[0, 2]), "sin(q)")

    def test_sym_transformation_Rz(self):
        Rz_sym = SymbolicTransformation.get_Rz()
        self.assertAlmostEqual(Rz_sym[2, 2], 1.0)
        self.assertEqual(str(Rz_sym[0, 0]), "cos(q)")
        self.assertEqual(str(Rz_sym[1, 1]), "cos(q)")
        self.assertEqual(str(Rz_sym[0, 1]), "-sin(q)")
        self.assertEqual(str(Rz_sym[1, 0]), "sin(q)")

    def test_sequence_of_sym_transforms(self):
        Tz = SymbolicTransformation("Tz")
        self.assertEqual(str(Tz[2, 3]), 'd_0')

        name = "l_1"
        Tl_1 = SymbolicTransformation("Tz", [name])
        self.assertEqual(str(Tl_1[2, 3]), name)

        TxTz = SymbolicTransformation("TxTz")
        self.assertEqual(str(TxTz[0, 3]), 'd_0')
        self.assertEqual(str(TxTz[2, 3]), 'd_1')

    def test_sym_transformation_to_numbers(self):
        T = SymbolicTransformation("TzTxTy")
        T_num = T.evaluate([1.0, 3.0, 4.0])

        T_num_real = np.eye(4)
        T_num_real[0, 3] = 3.0
        T_num_real[1, 3] = 4.0
        T_num_real[2, 3] = 1.0

        T_num = np.array(T_num.tolist(), dtype=float)

        self.assertTrue(np.allclose(T_num, T_num_real))

    def test_inverse_sym_transformation(self):

        T = SymbolicTransformation("Ty", ['y'])
        T_inv = T.inv()

        self.assertEqual(str(T_inv[1, 3]), '-y')

        T = SymbolicTransformation("RxTzTxRy")
        T_inv_inv = T.inv().inv()

        T_zeros = np.array(T_inv_inv.transformation - T.transformation,
                           dtype=float)
        self.assertTrue(np.allclose(np.zeros(4), T_zeros))

    def test_sym_transformation_substitutions(self):
        T = SymbolicTransformation("TxTyTz", ["x", "y", "z"])

        T.substitute([["x", 1], ["z", 2]])

        self.assertEqual(T[0, 3], 1)
        self.assertEqual(str(T[1, 3]), "y")
        self.assertEqual(T[2, 3], 2)

    def test_transformation_sequence_translations(self):
        variables = [1.0, 1.2, 0.7]
        seq = "TxTyTz"

        tf = Transformation(seq, variables)
        T = tf.transformation

        self.assertEqual(T[0, 3], 1.0)
        self.assertEqual(T[1, 3], 1.2)
        self.assertEqual(T[2, 3], 0.7)


if __name__ == '__main__':
    unittest.main()
