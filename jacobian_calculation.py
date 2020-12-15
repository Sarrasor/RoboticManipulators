import sympy as sp
from utils.jacobians import JacobianCalculator


def main():
    T_base = sp.eye(4)
    T_tool = sp.eye(4)

    variables = ['q_1', 'L_1', 'q_2', 'L_2']

    indices = [1, 0, 1, 0]
    sequence = "RzTxRzTx"

    jc = JacobianCalculator(sequence,
                            indices,
                            variables=variables,
                            T_base=T_base,
                            T_tool=T_tool,
                            f_of_t=False,
                            simplify=True)

    J_num = jc.calculate_numeric()
    J_skew = jc.calculate_scew()
    print("Numeric Jacobian calculation:")
    print()
    sp.pprint(J_num)
    print()
    print("Scew Jacobian calculation:")
    print()
    sp.pprint(J_skew)
    print()
    print("Difference between methods:")
    print()
    sp.pprint(J_skew - J_num)


if __name__ == '__main__':
    main()
