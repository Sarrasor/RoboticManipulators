"""
Symbolic transformation usage example
"""
from utils.robo_math import SymbolicTransformation as st


def main():
    T_ik = st("TxTyTzRzRyRx", ['x', 'y', 'z', 'alpha', 'theta', 'gamma'])
    T_0 = st("Tz", ['l_0']).inv() * T_ik * st("Tx", ["l_6"]).inv()

    T_012 = st("RzTzTxRyTxRyRyiTx",
               ["q_0", "l_1", "l_2", "q_1", "l_3", "q_2", "dq", "d"])

    T_345 = st("RyRxRyRx", ["dq", "q_3", "q_4", "q_5"])

    T_pos = T_0 * T_345.inv()
    T_rot = T_012.inv() * T_0

    print("T_012:")
    T_012.print()

    print("T_345:")
    T_345.print()

    print("T_pos:")
    T_pos.print()

    print("Position values:")
    print(f"X value: {T_pos.transformation[0, 3]}")
    print(f"Y value: {T_pos.transformation[1, 3]}")
    print(f"Z value: {T_pos.transformation[2, 3]}")
    print()
    print("Position equations:")
    print(f"X equation: {T_012.transformation[0, 3]}")
    print(f"Y equation: {T_012.transformation[1, 3]}")
    print(f"Z equation: {T_012.transformation[2, 3]}")

    print("T_rot:")
    T_rot.print()


if __name__ == '__main__':
    main()
