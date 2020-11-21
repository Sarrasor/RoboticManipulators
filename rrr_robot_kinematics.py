"""
RRR Robot FK and IK usage example
"""
import numpy as np

from robots.rrr_robot import RRRRobot


def main():
    np.set_printoptions(suppress=True)

    T_base = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T_tool = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    robot = RRRRobot(T_base=T_base, T_tool=T_tool)

    qs = [np.pi / 2, 0, 0.2]

    T = robot.forward_kinematics(qs, plot=False)

    print("Forward kinematics:")
    print(T)
    print()

    print("Real Qs:")
    print(qs)

    T_IK = T

    qs = robot.inverse_kinematics(T_IK, m=1, k=1)

    print()
    print("IK Qs:")
    print(qs)

    T_res = robot.forward_kinematics(qs, plot=True)

    print()
    print("After inverse kinematics:")
    print(T_res)


if __name__ == '__main__':
    main()
