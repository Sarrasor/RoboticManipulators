import numpy as np
import scipy.io
import sympy as sp
from utils.model_reducer import ModelReducer
from robots.fanuc import Fanuc165F
from utils.robo_math import Transformation as tf


def main():
    mat = scipy.io.loadmat('data/calibration_dataset.mat')
    qs = np.array(mat['q'], dtype=np.float)
    pts1 = np.array(mat['mA'], dtype=np.float)[:, :, np.newaxis]
    pts2 = np.array(mat['mB'], dtype=np.float)[:, :, np.newaxis]
    pts3 = np.array(mat['mC'], dtype=np.float)[:, :, np.newaxis]
    pts = np.concatenate((pts1, pts2, pts3), axis=1)

    # dx = 0.2
    # d = 0.0606
    # rot = 2 / 3 * sp.pi
    # # Pose of p1 wrt End-effector
    # E_T_p1 = tf.get_Tx(dx) * tf.get_Tz(d)
    # # Pose of p2 wrt End-effector
    # E_T_p2 = tf.get_Tx(dx) * tf.get_Rx(rot) * tf.get_Tz(d) * tf.get_Rx(-rot)
    # # Pose of p3 wrt End-effector
    # E_T_p3 = tf.get_Tx(dx) * tf.get_Rx(-rot) * tf.get_Tz(d) * tf.get_Rx(rot)

    # # Pose of Base wrt Faro
    # F_T_B = tf.get_Tx(6) * tf.get_Ty(-3) * tf.get_Tz(1)
    # F_T_B = F_T_B * tf.get_Rx(0.0) * tf.get_Ry(-0.0) * tf.get_Rz(0.0)

    # lengths_real = (0.346, 0.324, 0.312, 1.075, 0.225, 1.280, 0.215)

    # robot = Fanuc165F(lengths=lengths_real, save=False)
    # sigma = 0.0

    qs_offset = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
    qs_directions = np.array([1, 1, -1, -1, -1, -1])

    # pts1, pts2, pts3 = [], [], []
    # for q in qs:
    #     # Pose of End-effector wrt Base
    #     B_T_E = robot.forward_kinematics(q + qs_offset, plot=False)

    #     print("Synthetic is evaluated at:")
    #     print(q + qs_offset)
    #     print("Synthetic EF Position:")
    #     sp.pprint(B_T_E[:3, 3])
    #     print()

    #     # Coordinates of p_0 wrt Faro
    #     F_p_1 = (F_T_B * B_T_E * E_T_p1)[:3, 3]
    #     # Coordinates of p_1 wrt Faro
    #     F_p_2 = (F_T_B * B_T_E * E_T_p2)[:3, 3]
    #     # Coordinates of p_2 wrt Faro
    #     F_p_3 = (F_T_B * B_T_E * E_T_p3)[:3, 3]

    #     # Add noise
    #     noise_1 = np.random.normal(0, sigma, (3, 1))
    #     noise_2 = np.random.normal(0, sigma, (3, 1))
    #     noise_3 = np.random.normal(0, sigma, (3, 1))

    #     pts1.append(np.array(F_p_1, dtype=np.float) + noise_1)
    #     pts2.append(np.array(F_p_2, dtype=np.float) + noise_2)
    #     pts3.append(np.array(F_p_3, dtype=np.float) + noise_3)

    #     print(np.linalg.norm(pts1[-1] - pts2[-1]))
    #     print(np.linalg.norm(pts1[-1] - pts2[-1]))
    #     print(np.linalg.norm(pts3[-1] - pts1[-1]))
    #     print()

    lengths_nominal = [('d_1', 0.67), ('d_2', 0.312), ('d_3', 1.075),
                       ('d_5', 0.225), ('d_4', 1.280), ("d_6", 0.215)]
    variables = ['d_1', 'q_1', 'd_2', 'q_2', 'd_3', 'q_3',
                 'd_4', 'd_5', 'q_4', 'q_5', 'q_6', 'd_6']
    indices = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    sequence = "TzRzTxRyTxRyTxTzRxRyRxTx"

    mr = ModelReducer(sequence, indices,
                      variables=variables,
                      link_lengths=lengths_nominal)
    result_sequence, variables = mr.reduced_model()
    T_base, T_tools = mr

    Adp_sum = sp.zeros(15, 1)
    A_sum = sp.zeros(15, 15)
    for q, p_1, p_2, p_3 in zip(qs, pts1, pts2, pts3):
        q = np.multiply(q, qs_directions) + qs_offset
        R_robot, p_robot = mr.evaluate_at_nominal(q)

        dp_i = sp.Matrix([p_1 - p_robot,
                          p_2 - p_robot,
                          p_3 - p_robot])

        pairwise_1 = np.linalg.norm(p_1 - p_2)
        pairwise_2 = np.linalg.norm(p_3 - p_2)
        pairwise_3 = np.linalg.norm(p_1 - p_3)

        p_scew = tf.get_scew(p_robot[0], p_robot[1], p_robot[2]).T
        z = sp.Matrix([0, 0, 0])
        A_i = sp.Matrix([[sp.eye(3), p_scew, R_robot, z, z, z, z, z, z],
                         [sp.eye(3), p_scew, z, z, z, R_robot, z, z, z],
                         [sp.eye(3), p_scew, z, z, z, z, z, z, R_robot]])
        A_sum += A_i.T * A_i
        Adp_sum += A_i.T * dp_i

    est = (A_sum).inv() * Adp_sum

    p_base = est[:3]

    R_base = tf.get_Rx(est[3]) * tf.get_Ry(est[4]) * tf.get_Rz(est[5])
    R_base = R_base[:3, :3]
    p_1_tool = R_base.T * sp.Matrix([est[6], est[7], est[8]])
    p_2_tool = R_base.T * sp.Matrix([est[9], est[10], est[11]])
    p_3_tool = R_base.T * sp.Matrix([est[12], est[13], est[14]])

    sp.pprint(est)
    print("P base + Tz(d_1)")
    sp.pprint(p_base)
    print()
    print("R base:")
    sp.pprint(R_base)
    print()
    print("P1 tool + Tx(d_6):")
    sp.pprint(p_1_tool)
    print()
    print("P2 tool + Tx(d_6):")
    sp.pprint(p_2_tool)
    print()
    print("P3 tool + Tx(d_6):")
    sp.pprint(p_3_tool)
    print()

    print("Pairwise distances real:")
    print(pairwise_1)
    print(pairwise_2)
    print(pairwise_3)
    print()

    print("Pairwise distances estimated:")
    print((p_2_tool - p_1_tool).norm())
    print((p_3_tool - p_2_tool).norm())
    print((p_1_tool - p_3_tool).norm())


if __name__ == '__main__':
    main()
