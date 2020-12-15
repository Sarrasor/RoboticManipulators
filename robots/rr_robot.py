"""
RRRobot robot class definition
"""
import pickle
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from robots.robot import Robot
from utils.robo_math import SymbolicTransformation as st
from utils.plot_utils import TransformationPlotter


class RRRobot(Robot):
    """
    RRRobot manipulator Forward Kinematics (FK) and Inverse Kinematics(IK)
    calculator class

    Attributes:
        d (float): Length parameter from TxTz substitution
        dq (float): Angle parameter from TxTz substitution
        fk_data_path (pathlib.Path): Path to pickle forward kinematics data
            Used to speedup FK calculation
        ik_data_path (pathlib.Path): Path to pickle inverse kinematics data
            Used to speedup FK calculation
        ls (tuple): Lengths of links
        qs_lim_deg (tuple of tuples): Joint limits in degrees
        qs_lim_rad (tuple of tuples): Joint limits in radians
        T_base (4x4 array like): Transformation from world frame to the base
        T_tool (4x4 array like): Transformation from end-effector frame to the
            tool frame
    """

    qs_lim_deg = ((-360.0, 360.0), (-360.0, 360.0))

    def __init__(self, T_base=None, T_tool=None, lengths=None, save=True):
        """
        Prepares all necessary values and loads pickled matrices

        Args:
            T_base (None, optional): Transformation from the world frame
                to the base frame
            T_tool (None, optional): Transformation from the end-effector
                frame to the tool frame
        """
        self.set_transforms(T_base, T_tool)
        self.set_lengths(lengths)

        self._generate_value_pairs()
        self._calculate_limits_radians()

        self._save = save
        if self._save:
            self.fk_data_path = Path("robots/data/rr_forward_kinematics.pkl")
        self._precalculate_data()

        self._tp = TransformationPlotter()

    def set_lengths(self, lengths):
        if lengths is None:
            self._ls = (0.8, 0.8)
        else:
            self._ls = lengths

    def _generate_value_pairs(self):
        """
        Generates name-value tuples for sympy substitution
        """
        value_pairs = []
        for i in range(len(self._ls)):
            value_pairs.append((f"l_{i}", self._ls[i]))

        self._value_pairs = value_pairs

    def _calculate_limits_radians(self):
        """
        Converts joint limits from degrees to radians
        """
        self.qs_lim_rad = tuple(
            (np.deg2rad(x[0]), np.deg2rad(x[1])) for x in self.qs_lim_deg)

    def _precalculate_data(self):
        """
        Precalculates and pickles constant matrices
        """
        if self._save and self.fk_data_path.is_file():
            with open(self.fk_data_path, 'rb') as input:
                self._Ts = pickle.load(input)
        else:
            self._Ts = st("RzTxRzTx",
                          ['q_0', 'l_0', 'q_1', 'l_1'])

            self._Ts.substitute(self._value_pairs)

        # Save data
        if self._save and not self.fk_data_path.is_file():
            with open(self.fk_data_path, 'wb') as output:
                pickle.dump(self._Ts, output, pickle.HIGHEST_PROTOCOL)

    def forward_kinematics(self, q_values, plot=True):
        """
        Calculates forward kinematics of the tool pose given values of joints

        Args:
            q_values (list of float): Values of joints
            plot (bool, optional): Flag to plot the result

        Returns:
            4x4 np.ndarray: Homogeneous tool pose
        """
        qs_dict = {}
        for i in range(len(q_values)):
            qs_dict[sp.symbols(f"q_{i}")] = q_values[i]

        self._numeric_frames = []
        for frame in self._Ts.frames:
            self._numeric_frames.append(frame.evalf(subs=qs_dict))

        T = self.T_base * self._numeric_frames[-1] * self.T_tool

        if plot:
            self._show_fk()

        return np.array(T, dtype=np.float)

    def _show_fk(self):
        """
        Plots current pose of the robot
        """
        frames = [self.T_base]

        for frame, var in zip(self._numeric_frames, self._Ts.variables):
            if var[0] == 'q':
                frames.append(self.T_base * frame)

        frames.append(self.T_base * self._numeric_frames[-1])
        frames.append(frames[-1] * self.T_tool)

        self._tp.plot_numeric_frames(frames,
                                     axis_len=self._ls[0] / 8,
                                     margin=2,
                                     center=0,
                                     fixed_scale=True)

    def inverse_kinematics(self, T, m=1):
        """
        Calculates inverse kinematics joint values qs from pose T

        Args:
            T (4x4 array like): Homogeneous pose matrix
            m (int, optional): Elbow up flag. Should be -1 or 1
            k (int, optional): Square root sign flag. Should be -1 or 1

        Returns:
            np.ndarray: Joint values, corresponding to T
                or zeros in case of failure
        """
        if abs(m) != 1:
            print("[WARNING] m can only be -1 or 1. Defaulting to 1")
            m = 1

        T = sp.Matrix(T)
        T_0 = self.T_base.inv() * T * self.T_tool.inv()

        x, y = float(T_0[0, 3]), float(T_0[1, 3])

        arccos_numerator = x**2 + y**2 - self._ls[0]**2 - self._ls[1]**2
        arccos_denominator = 2.0 * self._ls[0] * self._ls[1]
        arccos = arccos_numerator / arccos_denominator

        # Check if the given position is reachable
        if abs(arccos) > 1:
            print("[INFO] The configuration is not reachable")
            return np.array([0.0, 0.0])

        q_1 = m * np.arccos(arccos)
        beta = np.arctan2(self._ls[1] * np.sin(m * q_1),
                          self._ls[0] + self._ls[1] * np.cos(q_1))
        q_0 = np.arctan2(y, x) - m * beta

        qs = np.array([q_0, q_1])

        out_of_limits = False
        for i in range(len(qs)):
            if qs[i] < self.qs_lim_rad[i][0] or qs[i] > self.qs_lim_rad[i][1]:
                print(f"[INFO] q_{i} = {np.rad2deg(qs[i]):.3f} "
                      f"(degrees) is out of limits")
                out_of_limits = True

        if out_of_limits:
            return np.array([0.0, 0.0])

        return qs

    def move_joints(self, qs):
        """
        Iteratively moves joints according to the trajectorties in qs

        Args:
            qs (np.ndarray): nx3 array, where n - is the number of points
        """
        plt.ion()
        Ts = []
        for q in qs:
            for T in Ts:
                self._tp.plot_position(T)

            T = self.forward_kinematics(q, plot=True)
            Ts.append(T)

            plt.pause(1e-9)
            self._tp.ax.cla()
        plt.ioff()
        for T in Ts[:-1]:
            self._tp.plot_position(T, show=False)
        self.forward_kinematics(qs[-1], plot=True)

    def move_via_points(self, pts):
        """
        Iteratively moves joints via cartesian points using IK

        Args:
            pts (np.ndarray): nx3 array, where n - is the number of points

        Returns:
            np.ndarray: Array of corresponding joint configurations
        """
        plt.ion()
        Ts = []
        qs = []
        for pt in pts:
            for T in Ts:
                self._tp.plot_position(T)

            self._tp.ax.scatter(
                pts[0][0],
                pts[0][1],
                pts[0][2],
                c='red',
                s=40,
                alpha=0.6,
            )

            self._tp.ax.scatter(
                pts[-1][0],
                pts[-1][1],
                pts[-1][2],
                c='red',
                s=40,
                alpha=0.6,
            )

            T_IK = np.array([
                [1, 0, 0, pt[0]],
                [0, 1, 0, pt[1]],
                [0, 0, 1, pt[2]],
                [0, 0, 0, 1]
            ])

            q = self.inverse_kinematics(T_IK)
            qs.append(q)
            T = self.forward_kinematics(q, plot=True)
            Ts.append(T)

            plt.pause(1e-9)
            self._tp.ax.cla()
        plt.ioff()
        for T in Ts[:-1]:
            self._tp.plot_position(T, show=False)
        self.forward_kinematics(q, plot=True)

        return np.array(qs).T
