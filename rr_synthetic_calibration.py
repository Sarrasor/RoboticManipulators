"""
Sample calibration of the RR manipulator with synthetic dataset
"""
import numpy as np
import sympy as sp
from utils.model_calibration import ModelCalibrator
from robots.rr_robot import RRRobot
from utils.robo_math import Transformation as tf


def main():
    # Pose of p1 wrt End-effector
    E_T_p1 = tf.get_Tx(0.1) * tf.get_Ty(0.1)
    # Pose of p2 wrt End-effector
    E_T_p2 = tf.get_Tx(0.1) * tf.get_Ty(-0.1)

    # Pose of Base wrt Faro
    F_T_B = tf.get_Tx(1.0) * tf.get_Ty(0.5) * tf.get_Tz(0.0)
    F_T_B *= tf.get_Rx(0.00) * tf.get_Ry(0.0) * tf.get_Rz(-0.0)

    lengths_real = (2.1, 1.0)

    robot = RRRobot(lengths=lengths_real, save=False)
    qs = np.array([
        [np.pi / 6, -np.pi / 2],
        [np.pi / 6, np.pi / 2],
        [np.pi / 6, np.pi * 0.555555556],
        [np.pi / 6 - np.pi, -np.pi / 2],
        [0.6523, 0.3],
        [-0.3, 0.2],
        [1.23, -0.117],
        [1.93, 0.157],
        [0.3, -1.1],
        [1.95, 1.24],
        [0.55, -0.4],
        [0.1, -3.2],
        [0.0, 0.0],
    ])

    sigma = 0.001

    pts0, pts1, pts2 = [], [], []
    for q in qs:
        # Pose of End-effector wrt Base
        B_T_E = robot.forward_kinematics(q, plot=False)
        # Coordinates of p_0 wrt Faro
        F_p_0 = (F_T_B * B_T_E)[:3, 3]
        # Coordinates of p_1 wrt Faro
        F_p_1 = (F_T_B * B_T_E * E_T_p1)[:3, 3]
        # Coordinates of p_2 wrt Faro
        F_p_2 = (F_T_B * B_T_E * E_T_p2)[:3, 3]

        # Add noise
        noise_0 = np.random.normal(0, sigma, (3, 1))
        noise_1 = np.random.normal(0, sigma, (3, 1))
        noise_2 = np.random.normal(0, sigma, (3, 1))

        pts0.append(np.array(F_p_0, dtype=np.float) + noise_0)
        pts1.append(np.array(F_p_1, dtype=np.float) + noise_1)
        pts2.append(np.array(F_p_2, dtype=np.float) + noise_2)

        # print(np.linalg.norm(pts0[-1] - pts1[-1]))
        # print(np.linalg.norm(pts1[-1] - pts2[-1]))
        # print(np.linalg.norm(pts2[-1] - pts0[-1]))
        # print()

    pts = np.concatenate((pts0, pts1, pts2), axis=1)

    # Nominal parameters
    offsets_nominal = np.array([0.0, 0.0])
    lengths_nominal = [('L_1', 2.0), ('L_2', 1.0)]

    # Create manipulator model
    sequence = "RzTxRzTx"
    variables = ['q_1', 'L_1', 'q_2', 'L_2']
    joint_indices = [1, 0, 1, 0]
    directions = np.array([1, 1])

    mc = ModelCalibrator(sequence,
                         joint_indices=joint_indices,
                         variables=variables,
                         link_lengths=lengths_nominal,
                         offsets=offsets_nominal,
                         directions=directions,
                         step=0.05)
    result_sequence, variables = mc.get_reduced_model()

    for i in range(5):
        T_base, T_tools = mc.estimate_tool_base(qs, pts)

        sp.pprint(T_base)
        print()
        sp.pprint(T_tools[0])
        print()
        sp.pprint(T_tools[1])
        print()
        sp.pprint(T_tools[2])

        mc.estimate_parameters(qs, pts, T_base, T_tools)

        mc.evaluate(qs, pts, T_base, T_tools)

    mc.optimize_with_base(qs, pts)

    mc.evaluate(qs, pts, T_base, T_tools)


if __name__ == '__main__':
    main()
