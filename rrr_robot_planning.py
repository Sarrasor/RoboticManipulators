"""
RRR Robot trajectory planning examples
"""
import numpy as np

from utils.trajectory_generator import TrajectoryGenerator
from robots.rrr_robot import RRRRobot
from rrr_robot_jacobians import get_symbolic_jacobian_inverse


def main():
    dq_max = 1
    ddq_max = 20
    dx_max = 1
    ddx_max = 10
    n = 50
    cf = 10

    tg = TrajectoryGenerator(dq_max, ddq_max, dx_max, ddx_max, control_freq=cf)

    # Polynomial joint trajectory
    q_0 = [[0.5, 0, 0], [-0.6, 0, 0], [0, 0, 0]]
    q_f = [[1.57, 0, 0], [0.5, 0, 0], [-2.0, 0, 0]]
    t_0, t_f = 0, 2
    qs = tg.generate_joint_poly_trajectory(q_0,
                                           q_f,
                                           t_f,
                                           t_0=t_0,
                                           n=n,
                                           plot=True)
    robot = RRRRobot()
    robot.move_joints(qs)

    # Trapezoidal joint trajectory
    q_0 = [0.0, 0.0, 0.0]
    q_f = [-0.9, -2.3, 1.2]

    qs = tg.generate_p2p_trajectory(q_0, q_f, n=n, plot=True)
    robot = RRRRobot()
    robot.move_joints(qs)

    # Trapezoidal cartesian linear trajectory
    p_1 = [1.0, 0.0, 2.0]
    p_2 = [np.sqrt(2) / 2, np.sqrt(2) / 2, 1.2]

    xs, dxs, ts = tg.generate_lin_trajectory(p_1, p_2, n=n, plot=True)
    robot = RRRRobot()
    qs = robot.move_via_points(xs)

    J_inv = get_symbolic_jacobian_inverse()
    tg.get_dq_from_dx(qs, dxs, J_inv, ts)


if __name__ == '__main__':
    main()
