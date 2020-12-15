import re

import numpy as np
import sympy as sp


class SymbolicTransformation():
    def __init__(self,
                 sequence_string,
                 variables=None,
                 f_of_t=None,
                 simplify=False):
        self._seq = sequence_string
        self._tokens = SymbolicTransformation._tokens_from_sequence(self._seq)
        self._tfs = [sp.eye(4)] * (len(self._tokens) + 1)
        self._token_to_transform = {
            'Tx': self.get_Tx,
            'Ty': self.get_Ty,
            'Tz': self.get_Tz,
            'Rx': self.get_Rx,
            'Ry': self.get_Ry,
            'Rz': self.get_Rz,
            'Txi': self.get_Tx_inv,
            'Tyi': self.get_Ty_inv,
            'Tzi': self.get_Tz_inv,
            'Rxi': self.get_Rx_inv,
            'Ryi': self.get_Ry_inv,
            'Rzi': self.get_Rz_inv,
            'Txd': self.get_Txd,
            'Tyd': self.get_Tyd,
            'Tzd': self.get_Tzd,
            'Rxd': self.get_Rxd,
            'Ryd': self.get_Ryd,
            'Rzd': self.get_Rzd,
        }

        if variables is None:
            self._variables = []
            t_index = 0
            r_index = 0

            for token in self._tokens:
                if token in self._token_to_transform.keys():
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

        self._f_of_t = {}
        if f_of_t is None:
            for var in self._variables:
                self._f_of_t[var] = False
        else:
            for var, val in zip(self._variables, f_of_t):
                self._f_of_t[var] = val

        self._generate_transformation(simplify=simplify)

    def _generate_transformation(self, simplify=False):
        for i, (token, var) in enumerate(zip(self._tokens, self._variables)):
            self._tfs[i + 1] = self._tfs[i] * self._from_token(token, var)
            if simplify:
                self._tfs[i + 1] = sp.simplify(self._tfs[i + 1])

    def _from_token(self, token, variable_name):
        try:
            f_of_t = False
            if variable_name in self._f_of_t:
                f_of_t = self._f_of_t[variable_name]
            return self._token_to_transform[token](variable_name, f_of_t)
        except Exception:
            raise ValueError("Unknown token")

    def valid_token(self, token):
        if token in self._token_to_transform:
            return True
        return False

    @staticmethod
    def _tokens_from_sequence(sequence):
        return [s for s in re.split("([A-Z][^A-Z]*)", sequence) if s]

    def __getitem__(self, i):
        return self._tfs[-1][i]

    def __mul__(self, other):
        new_sequence = self._seq + other._seq
        new_vars = self._variables + other._variables
        return SymbolicTransformation(new_sequence, new_vars)

    def __rmul__(self, other):
        new_sequence = other._seq + self._seq
        new_vars = other._variables + self._variables
        return SymbolicTransformation(new_sequence, new_vars)

    @property
    def transformation(self):
        return self._tfs[-1]

    @property
    def frames(self):
        return self._tfs[1:]

    @property
    def variables(self):
        return self._variables

    def substitute(self, var_value_pairs):
        var_value_tuples = []
        for name, value in var_value_pairs:
            var_value_tuples.append((sp.symbols(name), value))

        for i in range(len(self._tfs)):
            self._tfs[i] = self._tfs[i].subs(var_value_tuples)

    def evaluate(self, values):
        values_dict = {}
        for value, name in zip(values, self._variables):
            values_dict[sp.symbols(name)] = value

        return self.transformation.evalf(subs=values_dict)

    def evaluate_tuples(self, name_value_tuples):
        values_dict = {}
        for name, value in name_value_tuples:
            values_dict[sp.symbols(name)] = value

        return self.transformation.evalf(subs=values_dict)

    def inv(self):
        new_sequence = ""
        for token in self._tokens[::-1]:
            if token[-1] == 'i':
                new_sequence += token[:-1]
            else:
                new_sequence += (token + "i")

        new_vars = self._variables[::-1]
        return SymbolicTransformation(new_sequence, new_vars)

    def get_rotation(self):
        R = sp.Matrix(sp.eye(4))
        R[:3, :3] = self.transformation[:3, :3]
        return R

    def print(self):
        print()
        sp.pprint(self.transformation)
        print()

    @staticmethod
    def _get_symbol(symbol, f_of_t=False):
        if f_of_t:
            t = sp.Symbol('t')
            s = sp.Function(symbol)(t)
        else:
            s = sp.Symbol(symbol)
        return s

    @staticmethod
    def get_jacobian_column(J):
        return sp.Matrix([
            J[0, 3],
            J[1, 3],
            J[2, 3],
            J[2, 1],
            J[0, 2],
            J[1, 0],
        ])

    @staticmethod
    def get_scew(x, y, z, f_of_t=False):
        x = SymbolicTransformation._get_symbol(x, f_of_t)
        y = SymbolicTransformation._get_symbol(y, f_of_t)
        z = SymbolicTransformation._get_symbol(z, f_of_t)
        return sp.Matrix([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ])

    @staticmethod
    def get_inertia_matrix(index=0):
        Ixx = sp.symbols(f"Ixx_{index}")
        Iyy = sp.symbols(f"Iyy_{index}")
        Izz = sp.symbols(f"Izz_{index}")
        Ixy = sp.symbols(f"Ixy_{index}")
        Iyz = sp.symbols(f"Iyz_{index}")
        Ixz = sp.symbols(f"Ixz_{index}")
        return sp.Matrix([[Ixx, Ixy, Ixz],
                          [Ixy, Iyy, Iyz],
                          [Ixz, Iyz, Izz]])

    @staticmethod
    def get_Tx(symbol='x', f_of_t=False):
        x = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tx_inv(symbol='x', f_of_t=False):
        x = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, -x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Ty(symbol='y', f_of_t=False):
        y = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Ty_inv(symbol='y', f_of_t=False):
        y = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, -y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tz(symbol='z', f_of_t=False):
        z = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tz_inv(symbol='z', f_of_t=False):
        z = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -z],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Rx(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sp.cos(q), -sp.sin(q), 0.0],
            [0.0, sp.sin(q), sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Ry(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [sp.cos(q), 0.0, sp.sin(q), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sp.sin(q), 0.0, sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Rz(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [sp.cos(q), -sp.sin(q), 0.0, 0.0],
            [sp.sin(q), sp.cos(q), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def get_Rx_inv(symbol='q', f_of_t=False):
        return SymbolicTransformation.get_Rx(symbol).T

    @staticmethod
    def get_Ry_inv(symbol='q', f_of_t=False):
        return SymbolicTransformation.get_Ry(symbol).T

    @staticmethod
    def get_Rz_inv(symbol='q', f_of_t=False):
        return SymbolicTransformation.get_Rz(symbol).T

    @staticmethod
    def get_Txd(symbol='q', f_of_t=False):
        return sp.Matrix([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Tyd(symbol='q', f_of_t=False):
        return sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Tzd(symbol='q', f_of_t=False):
        return sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Rxd(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -sp.sin(q), -sp.cos(q), 0.0],
            [0.0, sp.cos(q), -sp.sin(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    @staticmethod
    def get_Ryd(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [-sp.sin(q), 0.0, sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-sp.cos(q), 0.0, -sp.sin(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    @staticmethod
    def get_Rzd(symbol='q', f_of_t=False):
        q = SymbolicTransformation._get_symbol(symbol, f_of_t)
        return sp.Matrix([
            [-sp.sin(q), -sp.cos(q), 0.0, 0.0],
            [sp.cos(q), -sp.sin(q), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])


class Transformation():
    def __init__(self, sequence_string, values):
        self._seq = sequence_string
        self._tokens = self._tokens_from_sequence(self._seq)
        self._tfs = [sp.eye(4)] * (len(self._tokens) + 1)
        self._token_to_transform = {
            'Tx': self.get_Tx,
            'Ty': self.get_Ty,
            'Tz': self.get_Tz,
            'Rx': self.get_Rx,
            'Ry': self.get_Ry,
            'Rz': self.get_Rz,
            'Txi': self.get_Tx_inv,
            'Tyi': self.get_Ty_inv,
            'Tzi': self.get_Tz_inv,
            'Rxi': self.get_Rx_inv,
            'Ryi': self.get_Ry_inv,
            'Rzi': self.get_Rz_inv,
            'Txd': self.get_Txd,
            'Tyd': self.get_Tyd,
            'Tzd': self.get_Tzd,
            'Rxd': self.get_Rxd,
            'Ryd': self.get_Ryd,
            'Rzd': self.get_Rzd,
        }

        if len(values) != len(self._tokens):
            raise ValueError("Transformation and values sizes do not match")

        self._values = values
        self._generate_transformation()

    def _generate_transformation(self, simplify=False):
        for i, (token, value) in enumerate(zip(self._tokens, self._values)):
            self._tfs[i + 1] = self._tfs[i] * self._from_token(token, value)

    def _from_token(self, token, value):
        try:
            return self._token_to_transform[token](value)
        except Exception:
            raise ValueError("Unknown token")

    def valid_token(self, token):
        if token in self._token_to_transform:
            return True
        return False

    @staticmethod
    def _tokens_from_sequence(sequence):
        return [s for s in re.split("([A-Z][^A-Z]*)", sequence) if s]

    @property
    def transformation(self):
        return self._tfs[-1]

    def inv(self):
        new_sequence = ""
        for token in self._tokens[::-1]:
            if token[-1] == 'i':
                new_sequence += token[:-1]
            else:
                new_sequence += (token + "i")

        new_values = self._values[::-1]
        return Transformation(new_sequence, new_values)

    @staticmethod
    def get_jacobian_column(J):
        return sp.Matrix([
            J[0, 3],
            J[1, 3],
            J[2, 3],
            J[2, 1],
            J[0, 2],
            J[1, 0],
        ])

    @staticmethod
    def get_scew(x, y, z):
        return sp.Matrix([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ])

    @staticmethod
    def get_angles(R):
        if R[2, 0] != 1:
            z = np.arctan2(R[2, 1], R[2, 2])
            x = np.arctan2(R[1, 0], R[0, 0])
            if R[2, 2] != 0:
                y = np.arctan2(R[2, 0], R[2, 2] / np.cos(z))
            else:
                if R[1, 1] != 0:
                    y = np.arctan2(-np.cos(x) * R[2, 0], R[0, 0])
                elif R[1, 0] != 0:
                    y = np.arctan2(-np.sin(x) * R[2, 0], R[1, 0])
        else:
            y = np.arcsin(R[2, 0])
            x = 0
            z = np.arctan2(R[0, 1], R[0, 2])
        return x, y, z

    @staticmethod
    def get_Tx(x):
        return sp.Matrix([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Ty(y):
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tz(z):
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tx_inv(x):
        return Transformation.get_Tx(-x)

    @staticmethod
    def get_Ty_inv(y):
        return Transformation.get_Ty(-y)

    @staticmethod
    def get_Tz_inv(z):
        return Transformation.get_Tz(-z)

    @staticmethod
    def get_Rx(q):
        return sp.Matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sp.cos(q), -sp.sin(q), 0.0],
            [0.0, sp.sin(q), sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Ry(q):
        return sp.Matrix([
            [sp.cos(q), 0.0, sp.sin(q), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sp.sin(q), 0.0, sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Rz(q):
        return sp.Matrix([
            [sp.cos(q), -sp.sin(q), 0.0, 0.0],
            [sp.sin(q), sp.cos(q), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def get_Rx_inv(q):
        return Transformation.get_Rx(q).T

    @staticmethod
    def get_Ry_inv(q):
        return Transformation.get_Ry(q).T

    @staticmethod
    def get_Rz_inv(q):
        return Transformation.get_Rz(q).T

    @staticmethod
    def get_Rx_from_degrees(q):
        return Transformation.get_Rx(np.deg2rad(q))

    @staticmethod
    def get_Ry_from_degrees(q):
        return Transformation.get_Ry(np.deg2rad(q))

    @staticmethod
    def get_Rz_from_degrees(q):
        return Transformation.get_Rz(np.deg2rad(q))

    @staticmethod
    def get_Txd(x):
        return sp.Matrix([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Tyd(y):
        return sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Tzd(z):
        return sp.Matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ])

    @staticmethod
    def get_Rxd(q):
        return sp.Matrix([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, -np.sin(q), -np.cos(q), 0.0],
            [0.0, np.cos(q), -np.sin(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    @staticmethod
    def get_Ryd(q):
        return sp.Matrix([
            [-np.sin(q), 0.0, np.cos(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-np.cos(q), 0.0, -np.sin(q), 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    @staticmethod
    def get_Rzd(q):
        return sp.Matrix([
            [-np.sin(q), -np.cos(q), 0.0, 0.0],
            [np.cos(q), -np.sin(q), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])


class Point3D():
    _x = 0.0
    _y = 0.0
    _z = 0.0

    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    def get_homogeneous(self):
        return np.array([[self._x], [self._y], [self._z], [1.0]])
