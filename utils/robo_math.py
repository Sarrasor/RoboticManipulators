import re

import numpy as np
import sympy as sp


class SymbolicTransformation():
    def __init__(self, sequence_string, variables=None):
        self._seq = sequence_string
        self._tokens = [s for s in re.split("([A-Z][^A-Z]*)", self._seq) if s]

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

        self._generate_transformation()

    def _generate_transformation(self):
        for i, (token, var) in enumerate(zip(self._tokens, self._variables)):
            self._tfs[i + 1] = self._tfs[i] * self._from_token(token, var)
            self._tfs[i + 1] = sp.simplify(self._tfs[i + 1])

    def _from_token(self, token, variable_name):
        try:
            return self._token_to_transform[token](variable_name)
        except Exception:
            raise ValueError("Unknown token")

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

    def print(self):
        print()
        sp.pprint(self.transformation)
        print()

    @staticmethod
    def get_Tx(symbol='x'):
        x = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tx_inv(symbol='x'):
        x = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, -x],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Ty(symbol='y'):
        y = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Ty_inv(symbol='y'):
        y = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, -y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tz(symbol='z'):
        z = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Tz_inv(symbol='z'):
        z = sp.symbols(symbol)
        return sp.Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -z],
            [0, 0, 0, 1],
        ])

    @staticmethod
    def get_Rx(symbol='q'):
        q = sp.symbols(symbol)
        return sp.Matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, sp.cos(q), -sp.sin(q), 0.0],
            [0.0, sp.sin(q), sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Ry(symbol='q'):
        q = sp.symbols(symbol)
        return sp.Matrix([
            [sp.cos(q), 0.0, sp.sin(q), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sp.sin(q), 0.0, sp.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Rz(symbol='q'):
        q = sp.symbols(symbol)
        return sp.Matrix([
            [sp.cos(q), -sp.sin(q), 0.0, 0.0],
            [sp.sin(q), sp.cos(q), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def get_Rx_inv(symbol='q'):
        return SymbolicTransformation.get_Rx(symbol).T

    @staticmethod
    def get_Ry_inv(symbol='q'):
        return SymbolicTransformation.get_Ry(symbol).T

    @staticmethod
    def get_Rz_inv(symbol='q'):
        return SymbolicTransformation.get_Rz(symbol).T


class Transformation():
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
            [0.0, np.cos(q), -np.sin(q), 0.0],
            [0.0, np.sin(q), np.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Ry(q):
        return sp.Matrix([
            [np.cos(q), 0.0, np.sin(q), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(q), 0.0, np.cos(q), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def get_Rz(q):
        return sp.Matrix([
            [np.cos(q), -np.sin(q), 0.0, 0.0],
            [np.sin(q), np.cos(q), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

    @staticmethod
    def get_Rx_from_degrees(q):
        return Transformation.get_Rx(np.deg2rad(q))

    @staticmethod
    def get_Ry_from_degrees(q):
        return Transformation.get_Ry(np.deg2rad(q))

    @staticmethod
    def get_Rz_from_degrees(q):
        return Transformation.get_Rz(np.deg2rad(q))


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
