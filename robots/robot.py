"""
Abstact Robot class definition
"""
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
