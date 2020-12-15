"""
Fanuc FK and IK usage example
"""
import numpy as np

from robots.fanuc import Fanuc165F


def main():
    np.set_printoptions(suppress=True)

    T_base = np.array([
        [1, 0, 0, 500],
        [0, 1, 0, 300],
        [0, 0, 1, -100],
        [0, 0, 0, 1]
    ])

    T_tool = np.array([
        [1, 0, 0, 100],
        [0, 1, 0, 0],
        [0, 0, 1, -300],
        [0, 0, 0, 1]
    ])

    robot = Fanuc165F(T_base=T_base, T_tool=T_tool)

    qs = [-0.485, 0.5, -0.23, 0.23, 0.21, 0.213]

    T = robot.forward_kinematics(qs, plot=False)

    print("Forward kinematics:")
    print(T)
    print()

    print("Real Qs:")
    print(qs)

    T_IK = T

    # T_IK = np.array([
    #     [1, 0, 0, 2000],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 900],
    #     [0, 0, 0, 1]
    # ])

    qs = robot.inverse_kinematics(T_IK, m=-1, w=1)

    print()
    print("IK Qs:")
    print(qs)

    T_res = robot.forward_kinematics(qs, plot=True)

    print()
    print("After inverse kinematics:")
    print(T_res)


if __name__ == '__main__':
    main()
