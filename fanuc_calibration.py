"""
Fanuc calibration example
"""
import scipy.io
import numpy as np
import sympy as sp

from utils.model_calibration import ModelCalibrator


def main():
    # Extract the dataset
    mat = scipy.io.loadmat('data/calibration_dataset.mat')
    qs = np.array(mat['q'], dtype=np.float)
    pts1 = np.array(mat['mA'], dtype=np.float)[:, :, np.newaxis]
    pts2 = np.array(mat['mB'], dtype=np.float)[:, :, np.newaxis]
    pts3 = np.array(mat['mC'], dtype=np.float)[:, :, np.newaxis]

    # Convert to millimiters
    pts = np.concatenate((pts1, pts2, pts3), axis=1) * 1000

    # Reduce the number of points
    qs_input = qs
    pts_input = pts

    # Initialize nominal parameters
    offsets_nominal = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
    lengths_nominal = [('d_1', 670), ('d_2', 312), ('d_3', 1075),
                       ('d_5', 225), ('d_4', 1280), ("d_6", 215)]

    # Create manipulator model
    sequence = "TzRzTxRyTxRyTxTzRxRyRxTx"
    variables = ['d_1', 'q_1', 'd_2', 'q_2', 'd_3', 'q_3',
                 'd_4', 'd_5', 'q_4', 'q_5', 'q_6', 'd_6']
    joint_indices = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    directions = np.array([1, 1, -1, -1, -1, -1])

    mc = ModelCalibrator(sequence,
                         joint_indices=joint_indices,
                         variables=variables,
                         link_lengths=lengths_nominal,
                         offsets=offsets_nominal,
                         directions=directions)
    result_sequence, variables = mc.get_reduced_model()

    T_base_0 = sp.Matrix([[1, 0, 0, -563.145620015451],
                          [0, 1, 0, -559.28130331986],
                          [0, 0, 1, 554.98685687043],
                          [0, 0, 0, 1]])

    T_tool_0_1 = sp.Matrix([[1, 0, 0, 416.687065414667],
                            [0, 1, 0, -17.0660344931335],
                            [0, 0, 1, 59.6890106614009],
                            [0, 0, 0, 1]])

    T_tool_0_2 = sp.Matrix([[1, 0, 0, 420.667529952858],
                            [0, 1, 0, 57.1117905464326],
                            [0, 0, 1, -15.7733691549466],
                            [0, 0, 0, 1]])

    T_tool_0_3 = sp.Matrix([[1, 0, 0, 420.483792669375],
                            [0, 1, 0, -43.7121193657286],
                            [0, 0, 1, -42.4708914215651],
                            [0, 0, 0, 1]])

    T_tools_0 = [T_tool_0_1, T_tool_0_2, T_tool_0_3]

    print("Initial error")
    mc.evaluate(qs, pts, T_base_0, T_tools_0)

    np.set_printoptions(precision=3, suppress=True)

    for i in range(1):
        T_base, T_tools = mc.estimate_tool_base(qs_input, pts_input)

        print("Tbase")
        sp.pprint(T_base)
        print()
        print("Ttool 1")
        sp.pprint(T_tools[0])
        print()
        print("Ttool 2")
        sp.pprint(T_tools[1])
        print()
        print("Ttool 3")
        sp.pprint(T_tools[2])

        mc.estimate_parameters(qs_input, pts_input, T_base, T_tools)

        print("After parameter calibration:")
        mc.evaluate(qs, pts, T_base, T_tools)

    mc.optimize(qs_input, pts_input, T_base, T_tools)

    print("After gradient step:")
    mc.evaluate(qs, pts, T_base, T_tools)


if __name__ == '__main__':
    main()
