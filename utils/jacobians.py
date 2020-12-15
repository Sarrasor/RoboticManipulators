"""
JacobianCalculator class definition
"""
import sympy as sp
from utils.robo_math import SymbolicTransformation as st
from utils.robo_math import Transformation as tf


class JacobianCalculator():
    """
    Jacobian calculation class

    Attributes:
        simplify (bool): Flag to apply symbolic simplification
        T_base (sp.Matrix): Transformation from the world frame to the
            base frame
        T_tool (sp.Matrix): Transformation from the end-effector frame
            to the tool frame
    """

    def __init__(self,
                 sequence_string,
                 joint_indices,
                 f_of_t=True,
                 variables=None,
                 T_base=None,
                 T_tool=None,
                 simplify=False):
        """
        Prepares all necessary data

        Args:
            sequence_string (str): Transformation sequence
            joint_indices (list of bool): Mask list with True on elements with
             that are not constant
            f_of_t (list of bool or bool): Mask list with True on
                elements with that are functions of time. If True is provided,
                will use joint indices to create the list automatically
            variables (None, optional): List with names of variables
            T_base (None, optional): Transformation from the world frame
                to the base frame
            T_tool (None, optional): Transformation from the end-effector
                frame to the tool frame
            simplify (bool, optional): Flag to apply symbolic simplification

        Raises:
            ValueError: Error when dimensions do not match or incorrect
                sequence string is provided
        """
        self._seq = sequence_string
        self._tokens = st._tokens_from_sequence(self._seq)
        self._seq_numeric = []
        self._seq_skew = []
        self._vars_skew = []
        self._ind = joint_indices
        self._validator = st('')
        self.simplify = simplify

        self.set_transforms(T_base, T_tool)

        if len(self._tokens) != len(self._ind):
            raise ValueError("Size of indices does not match")

        elif type(f_of_t) == bool:
            self._f_of_t = [False] * len(self._ind)
            if f_of_t:
                self._f_of_t = self._ind
        else:
            self._f_of_t = f_of_t

        # Generate variable names if necessary
        if variables is None:
            self._variables = []
            t_index = 0
            r_index = 0

            for token, index in zip(self._tokens, self._ind):
                if self._validator.valid_token(token):
                    if token.startswith('T'):
                        self._variables.append(f"d_{t_index}")
                        t_index += 1
                    elif token.startswith('R'):
                        self._variables.append(f"q_{r_index}")
                        r_index += 1
                else:
                    raise ValueError("Unknown transformation")
        else:
            if len(variables) != len(self._tokens):
                raise ValueError("Transformation and var sizes do not match")

            self._variables = variables

        # Generate sequences for jacobian calculations
        for i, (token, index) in enumerate(zip(self._tokens, self._ind)):
            if index:
                self._seq_numeric.append(''.join(
                    self._tokens[:i] + [f"{token}d"] + self._tokens[i + 1:]))
                self._seq_skew.append(''.join(self._tokens[:i]))

        self._seq_skew.append(''.join(self._tokens))
        self._seq_numeric.append(''.join(self._tokens))

        for seq in self._seq_skew:
            n = len(st._tokens_from_sequence(seq))
            self._vars_skew.append(self._variables[:n])

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

    def calculate_numeric(self, T_base=None, T_tool=None):
        """
        Calculates Jacobian matrix using the matrix differentiation method

        Args:
            T_base (None, optional): Base transformation matrix
            T_tool (None, optional): Tool transformation matrix

        Returns:
            sp.Matrix: Calculated Jacobian matirx
        """
        if T_base is None:
            T_base = self.T_base

        if T_tool is None:
            T_tool = self.T_tool

        T_robot = st(self._seq_numeric[-1],
                     self._variables,
                     f_of_t=self._f_of_t).transformation
        R = T_base * T_robot * T_tool
        R[0, 3] = R[1, 3] = R[2, 3] = 0.0
        Rt = R.T

        Jrs = []
        for seq in self._seq_numeric[:-1]:
            T_diff = st(seq,
                        self._variables,
                        f_of_t=self._f_of_t).transformation
            Jr = T_base * T_diff * T_tool * Rt
            Jrs.append(st.get_jacobian_column(Jr).T)

        if self.simplify:
            res = sp.simplify(sp.Matrix(Jrs).T)
        else:
            res = sp.Matrix(Jrs).T

        return res

    def calculate_numerically(self, values, T_base=None, T_tool=None):
        if T_base is None:
            T_base = self.T_base

        if T_tool is None:
            T_tool = self.T_tool

        T_robot = tf(self._seq_numeric[-1], values).transformation
        R = T_base * T_robot * T_tool
        R[0, 3] = R[1, 3] = R[2, 3] = 0.0
        Rt = R.T

        Jrs = []
        for sequence in self._seq_numeric[:-1]:
            T_diff = tf(sequence, values).transformation
            Jr = T_base * T_diff * T_tool * Rt
            Jrs.append(tf.get_jacobian_column(Jr).T)

        return sp.Matrix(Jrs).T

    def calculate_scew(self, T_base=None, T_tool=None):
        """
        Calculates Jacobian matrix using the scew theory method

        Args:
            T_base (None, optional): Base transformation matrix
            T_tool (None, optional): Tool transformation matrix

        Returns:
            sp.Matrix: Calculated Jacobian matirx
        """
        if T_base is None:
            T_base = self.T_base

        if T_tool is None:
            T_tool = self.T_tool

        w_T_n = st(self._seq_skew[-1],
                   self._vars_skew[-1],
                   f_of_t=self._f_of_t).transformation
        w_T_n = T_base * w_T_n * T_tool

        O_n = w_T_n[:3, 3]
        w_T_0 = st(self._seq_skew[0],
                   self._vars_skew[0],
                   f_of_t=self._f_of_t).transformation
        w_T_0 = T_base * w_T_0

        Ts, Js = [w_T_0], []
        for k in range(len(self._seq_skew) - 1):
            T = st(self._seq_skew[k + 1],
                   self._vars_skew[k + 1],
                   f_of_t=self._f_of_t).transformation
            Ts.append(T_base * T)

            # Get axis index from char
            axis_transform = self._seq_skew[k + 1][len(self._seq_skew[k])]
            axis_char = self._seq_skew[k + 1][len(self._seq_skew[k]) + 1]
            axis_index = ord(axis_char) - ord('x')
            U_k_prev = Ts[k][:3, axis_index]
            O_k_prev = Ts[k][:3, 3]

            if axis_transform == 'R':
                Js.append(
                    sp.Matrix([U_k_prev.cross(O_n - O_k_prev), U_k_prev]).T)
            elif axis_transform == 'T':
                Js.append(
                    sp.Matrix([U_k_prev, 0, 0, 0]).T)

        if self.simplify:
            res = sp.simplify(sp.Matrix(Js).T)
        else:
            res = sp.Matrix(Js).T

        return res
