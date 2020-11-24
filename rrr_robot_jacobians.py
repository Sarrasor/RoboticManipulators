"""
RRR Robot Jacobian calculation
"""
import pickle
from pathlib import Path

import sympy as sp

from utils.jacobians import JacobianCalculator

data_path = Path("data/rrr_jacobian.pkl")


def get_symbolic_jacobian_inverse():
    if data_path.is_file():
        with open(data_path, 'rb') as input:
            J_inv = pickle.load(input)
    else:
        T_base = sp.eye(4)
        T_tool = sp.eye(4)

        variables = ['q_0', 'l_0', 'q_1', 'l_1', 'q_2', 'l_2']

        indices = [1, 0, 1, 0, 1, 0]
        sequence = "RzTzRyTxRyTx"

        jc = JacobianCalculator(sequence,
                                indices,
                                variables=variables,
                                T_base=T_base,
                                T_tool=T_tool,
                                simplify=False)

        ls = (1.0, 1.0, 1.0)

        value_pairs = []
        for i in range(len(ls)):
            value_pairs.append((f"l_{i}", ls[i]))

        J_inv = jc.calculate_numeric()[:3, :3].inv()
        J_inv = sp.simplify(J_inv.subs(value_pairs))

        with open(data_path, 'wb') as output:
            pickle.dump(J_inv, output, pickle.HIGHEST_PROTOCOL)

    return J_inv


def main():
    T_base = sp.eye(4)
    T_tool = sp.eye(4)

    variables = ['q_0', 'l_0', 'q_1', 'l_1', 'q_2', 'l_2']

    indices = [1, 0, 1, 0, 1, 0]
    sequence = "RzTzRyTxRyTx"

    jc = JacobianCalculator(sequence,
                            indices,
                            variables=variables,
                            T_base=T_base,
                            T_tool=T_tool,
                            simplify=False)

    J_num = sp.simplify(jc.calculate_numeric())
    print("Numeric Jacobian:")
    print()
    sp.pprint(J_num)
    print()
    print("Inverse translation part of the Jacobian:")
    J_transl = J_num[:3, :3]
    print()
    sp.pprint(sp.simplify(J_transl.inv()))
    print()


if __name__ == '__main__':
    main()
