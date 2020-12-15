"""
Abstact Robot class definition
"""
import sympy as sp

from abc import ABC, abstractmethod


class Robot(ABC):
    """
    Robot base class

    Attributes:
        epsilon (float): Minimum difference to treat floats equal
    """

    epsilon = 1e-5

    @abstractmethod
    def forward_kinematics(self, q_values):
        """
        Calculates forward kinematics pose T given values of joints

        Args:
            q_values (list of float): Values of joints
        """
        pass

    @abstractmethod
    def inverse_kinematics(self, T):
        """
        Calculates inverse kinematics joint values qs from pose T

        Args:
            T (4x4 array like): Homogeneous pose matrix
        """
        pass

    def set_transforms(self, T_base=None, T_tool=None):
        """
        Updates base and tool transformations

        Args:
            T_base (None, optional): Transformation from the world frame
                to the base frame
            T_tool (None, optional): Transformation from the end-effector
                frame to the tool frame
        """
        if T_base is None:
            self.T_base = sp.eye(4)
        else:
            self.T_base = sp.Matrix(T_base)

        if T_tool is None:
            self.T_tool = sp.eye(4)
        else:
            self.T_tool = sp.Matrix(T_tool)

    def set_lengths(self, lengths):
        if lengths is None:
            self._ls = (0.8, 0.8)
        else:
            self._ls = lengths
