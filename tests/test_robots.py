import unittest

import numpy as np

from robots.fanuc import Fanuc165F


class TestRobotsModule(unittest.TestCase):
    def test_fanuc_ifk(self):
        print()
        robot = Fanuc165F()

        T = robot.forward_kinematics([0, 0, 0, 0, 0, 0], plot=False)
        T_real = np.array([
            [1, 0, 0, 1807],
            [0, 1, 0, 0],
            [0, 0, 1, 1970],
            [0, 0, 0, 1],
        ], dtype=float)

        self.assertTrue(np.all((T_real - T) <= robot.epsilon))

        T = robot.forward_kinematics([np.pi / 2, 0, 0, 0, 0, 0], plot=False)
        T_real = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 1807],
            [0, 0, 1, 1970],
            [0, 0, 0, 1],
        ], dtype=float)

        self.assertTrue(np.all((T_real - T) <= robot.epsilon))

    def test_fanuc_ik(self):
        robot = Fanuc165F()

        T = robot.forward_kinematics([1.570796, -0.28, 0.1, 0, 0.2, 0],
                                     plot=False)
        qs = robot.inverse_kinematics(T, m=1)
        qs_1 = robot.inverse_kinematics(T, m=-1)
        T_inv = robot.forward_kinematics(qs, plot=False)
        T_inv_1 = robot.forward_kinematics(qs_1, plot=False)

        self.assertTrue(np.all((T_inv - T) <= robot.epsilon))
        self.assertTrue(np.all((T_inv_1 - T) <= robot.epsilon))

    def test_fanuc_ik_rotation_singulartiy(self):
        robot = Fanuc165F()

        T = robot.forward_kinematics([0, 0, 0, 0, 0, 0], plot=False)
        qs = robot.inverse_kinematics(T, m=1)

        self.assertTrue(np.all((qs) <= robot.epsilon))

    def test_fanuc_ik_unreachable(self):
        robot = Fanuc165F()

        T = np.array([
            [1, 0, 0, 100000],
            [0, 1, 0, 2882],
            [0, 0, 1, 895],
            [0, 0, 0, 1],
        ], dtype=float)
        qs = robot.inverse_kinematics(T, m=1)

        self.assertTrue(np.all((qs) <= robot.epsilon))

    def test_fanuc_ik_q_limit(self):
        robot = Fanuc165F()

        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 100],
            [0, 0, 0, 1],
        ], dtype=float)
        qs = robot.inverse_kinematics(T, m=1)

        self.assertTrue(np.all((qs) <= robot.epsilon))
