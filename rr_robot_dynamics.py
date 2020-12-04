import numpy as np
from utils.dynamics_generator import DynamicsGenerator
from robots.rr_robot import RRRobot
from utils.desired_trajectory_generator import batman_trajectory
from utils.desired_trajectory_generator import line_trajectory
from utils.desired_trajectory_generator import ellipse_trajectory
from utils.desired_trajectory_generator import func_from_array


def get_ufunc(us, T):
    return func_from_array(us, T)


def get_desired_trajectory():
    # points = line_trajectory()
    points = ellipse_trajectory()
    # points = batman_trajectory()
    robot = RRRobot()

    qs = []
    for p in points:
        T_IK = np.array([
            [1, 0, 0, p[0]],
            [0, 1, 0, p[1]],
            [0, 0, 1, p[2]],
            [0, 0, 0, 1]
        ])
        qs.append(robot.inverse_kinematics(T_IK))
    qs = np.array(qs)

    # Smooth out qs
    N = 11
    qs[:, 0] = np.convolve(qs[:, 0], np.ones((N,)) / N)[(N - 1):]
    qs[:, 1] = np.convolve(qs[:, 1], np.ones((N,)) / N)[(N - 1):]
    qs = qs[:-10]

    # Generate dqs and ddqs
    dt = 0.005
    dqs = np.diff(qs, axis=0) / dt
    ddqs = np.diff(dqs, axis=0) / dt
    qs = qs[:-2]
    dqs = dqs[:-1]
    T = dt * len(qs)

    # Pack arrays as functions
    q_func = func_from_array(qs, T)
    dq_func = func_from_array(dqs, T)
    ddq_func = func_from_array(ddqs, T)

    return q_func, dq_func, ddq_func, T, dt


def main():
    # Generate RR Robot model
    variables = ['theta_1', 'L', 'L', 'theta_2', 'L', 'L']
    joint_indices = [1, 0, 0, 1, 0, 0]
    cm_indices = [1, 0, 0, 0, 1, 0]
    sequence = "RzTxTxRzTxTx"
    gravity_axis = "y"
    parameters = [('m_0', 3.0),
                  ('m_1', 4.0),
                  ('L', 0.4),
                  ('Izz_0', 1),
                  ('Izz_1', 2),
                  ('g', 9.81)]

    idg = DynamicsGenerator(sequence,
                            cm_indices,
                            joint_indices,
                            gravity_axis=gravity_axis,
                            print_equation=True,
                            variables=variables,
                            method='newton-euler')

    q_func, dq_func, ddq_func, T, dt = get_desired_trajectory()

    # Solve inverse dynamics problem
    us = idg.generate_control(q_func,
                              dq_func,
                              ddq_func,
                              parameters,
                              T=T,
                              dt=dt,
                              plot=True)

    # No control
    # us = np.zeros_like(us)

    fdg = DynamicsGenerator(sequence,
                            cm_indices,
                            joint_indices,
                            gravity_axis=gravity_axis,
                            print_equation=True,
                            variables=variables,
                            method='lagrange')

    # Initial values
    q_0 = q_func(0)
    dq_0 = dq_func(0)

    # Solve forward dynamics problem
    qs = fdg.generate_trajectory(q_0,
                                 dq_0,
                                 get_ufunc(us, T),
                                 parameters,
                                 T=T,
                                 dt=dt,
                                 plot=True)

    # Reduce qs size for faster plotting
    reduce_size = 3
    for i in range(reduce_size):
        qs = qs[np.mod(np.arange(len(qs)), 2) != 0]

    # Plot qs
    robot = RRRobot()
    robot.move_joints(qs)


if __name__ == '__main__':
    main()
