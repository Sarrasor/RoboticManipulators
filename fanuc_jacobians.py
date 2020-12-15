import numpy as np
import sympy as sp
from utils.jacobians import JacobianCalculator


def main():
    T_base = sp.eye(4)
    T_tool = sp.eye(4)

    variables = ['l_0', 'q_0', 'l_1', 'l_2', 'q_1',
                 'l_3', 'q_2', 'l_4', 'l_5', 'q_3',
                 'q_4', 'q_5', 'l_6']

    indices = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    sequence = "TzRzTzTxRyTxRyTzTxRxRyRxTx"

    jc = JacobianCalculator(sequence,
                            indices,
                            variables=variables,
                            T_base=T_base,
                            T_tool=T_tool,
                            f_of_t=False,
                            simplify=False)

    J_num = jc.calculate_numeric()
    J_skew = jc.calculate_scew()
    print("Numeric Jacobian calculation:")
    sp.pprint(J_num)
    print()
    print("Scew Jacobian calculation:")
    sp.pprint(J_skew)
    print()
    print("Difference between methods:")
    sp.pprint(sp.simplify(J_skew - J_num))

    ls = (346.0, 324.0, 312.0, 1075.0, 225.0, 1280.0, 215.0)

    # No singularity
    qs = (0.3, -0.2, 0.1, 0.23, 1.57, -0.2)
    # Rotation singularity
    qs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Full extension singularity
    qs = (np.pi, -np.pi / 2, 0.0, 0, np.pi, 0.0)
    # Ballerina singularity
    qs = (0, -0.812734284248, -1.526091868267, np.pi, 2.37356282787, np.pi)

    values_dict = {}
    for i in range(len(ls)):
        values_dict[sp.symbols(f"l_{i}")] = ls[i]
    for i in range(len(qs)):
        values_dict[sp.symbols(f"q_{i}")] = qs[i]

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print("Determinant of the Jacobian:")
    print(J_num.evalf(subs=values_dict).det())


if __name__ == '__main__':
    main()
