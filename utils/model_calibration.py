"""
Definition of ModelCalibrator class
"""
import re
import numpy as np
import sympy as sp
from tqdm import tqdm
from scipy.optimize import least_squares

from utils.robo_math import Transformation as tf
from utils.robo_math import SymbolicTransformation as st
from utils.jacobians import JacobianCalculator


class ModelCalibrator(object):
    """
    Creates irreducable model for a transformation sequence
    and estimates T_base, T_tool and geometric parameters from dataset of
    joint configurations and corresponding end-effector measurements
    """

    def __init__(self, sequence_string, joint_indices, variables,
                 link_lengths, offsets=None, directions=None, step=0.001):
        """
        Initialization

        Args:
            sequence_string (str): Transformation sequence
            joint_indices (list of bool): Mask list with True on elements with
                that are not constant
            variables (list of str): List with names of variables
            link_lengths (list of (str, float)): Name-value pairs of link
                lengths. [('d_1', 10.0), ('d_2', 30.0)], for example
            offsets (None, optional): Array of joint offsets
            directions (None, optional): Array that defines rotation direction
                [1, -1, 1], for example (1 - counterclockwise -1 - clockwise)
            step (float, optional): Parameter update step. Blends current
                parameters with estimated as: (1 - s) * old + s * new

        Raises:
            ValueError: Error when dimensions do not match or incorrect
                sequence string is provided
        """
        self._seq = sequence_string
        self._tokens = st._tokens_from_sequence(self._seq)
        self._ind = joint_indices
        self._validator = st('')
        self._jc = None
        self._parameters = None
        self._base_est = None
        self._T_base = None
        self._T_tools = None
        self._step = step
        self._no_model_msg = """
        Please, create the model first with ModelCalibrator.get_reduced_model()
        """

        self._base_iteration = 0
        self._params_iteration = 0

        self.set_link_lengths(link_lengths)
        self.set_joint_offsets(offsets)
        self.set_joint_directions(directions)

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

        self._j_seq, self._j_var, self._j_ind = [], [], []
        # Generate sequences of transforms for joint-link pairs
        indices = [i for i, x in enumerate(self._ind + [1]) if x == 1]
        for i, j in zip(indices, indices[1:]):
            self._j_seq.append(''.join(self._tokens[i:j]))
            self._j_var.append(self._variables[i:j])
            self._j_ind.append(self._ind[i:j])

        # Determine what variables will go to the base and the tool
        first_j = self._ind.index(1)
        last_j = len(self._ind) - 1 - self._ind[::-1].index(1)

        self._to_base = self._tokens[:first_j]
        self._to_tool = self._tokens[last_j + 1:]
        self._to_base_vars = self._variables[:first_j]
        self._to_tool_vars = self._variables[last_j + 1:]

    def set_link_lengths(self, link_lengths):
        if link_lengths is None:
            self._link_lengths = np.ones(len(self._ind) - sum(self._ind))
        else:
            self._link_lengths = link_lengths

    def set_joint_offsets(self, offsets):
        if offsets is None:
            self.qs_offsets = np.zeros(sum(self._ind))
        else:
            self.qs_offsets = offsets

    def set_joint_directions(self, directions):
        if directions is None:
            self.qs_directions = np.ones(sum(self._ind))
        else:
            self.qs_directions = directions

    def get_reduced_model(self):
        """
        Creates reduced model from the input transformation sequence

        Returns:
            str, list of str: Resultant reduced model sequence
                and corresponding variables
        """
        # Some useful strings
        xyz, base_seq = "xyz", "TxTyTzRxRyRz"

        # Joint-Link sequences
        jl_sequences = []
        for seq, is_joint, var in zip(self._j_seq, self._j_var, self._j_ind):
            if seq[0] == 'R':
                joint_sequence = f"R{seq[1]}"
                link_sequence = re.sub(f"R{seq[1]}|T{seq[1]}", '', base_seq)
                jl_sequences.extend([joint_sequence + link_sequence])
            elif seq[0] == 'T':
                joint_sequence = f"T{seq[1]}"
                link_sequence = re.sub(f"R{seq[1]}", '', base_seq[6:])
                jl_sequences.extend([joint_sequence + link_sequence])

        for j in range(1, len(jl_sequences)):
            cur_seq = jl_sequences[j]
            prev_seq = jl_sequences[j - 1]

            # If consequtive RR or TT
            if cur_seq[0] == prev_seq[0]:
                if cur_seq[0] == 'R':
                    # If orthogonal
                    if cur_seq[1] != prev_seq[1]:
                        to_subs = f"R{cur_seq[1]}"
                        jl_sequences[j - 1] = re.sub(to_subs, '', prev_seq)
                    # If parallel
                    else:
                        k = 1
                        while j - k >= 0:
                            j_seq = jl_sequences[j - k]
                            if j_seq[0] == 'T':
                                k += 1
                                continue

                            j_link = self._j_seq[k][2:]
                            if len(j_link) > 0:
                                j_link = j_link[1]
                            axis = re.sub(f"{j_seq[1]}|{j_link}", '', xyz)
                            if len(axis) > 1:
                                axis = axis[0]
                            to_subs = f"T{axis}"
                            jl_sequences[j - k] = re.sub(to_subs, '', j_seq)
                            break
                elif cur_seq[0] == 'T':
                    # If orthogonal
                    if cur_seq[1] != prev_seq[1]:
                        to_subs = f"T{cur_seq[1]}"
                        jl_sequences[j - 1] = re.sub(to_subs, '', prev_seq)
                    # If parallel
                    else:
                        k = 1
                        while j - k >= 0:
                            j_seq = jl_sequences[j - k]
                            if j_seq[0] == 'T' or j_seq[1] == cur_seq[1]:
                                k += 1
                                continue
                            to_subs = f"T{cur_seq[1]}"
                            jl_sequences[j - k] = re.sub(to_subs, '', j_seq)
                            break

        # Move translations to tool, considering small errors in rotations
        barriers = set()
        for j in range(len(jl_sequences) - 1, 0, -1):
            cur_seq = jl_sequences[j]
            prev_seq = jl_sequences[j - 1]
            barriers.add(cur_seq[1])
            if len(barriers) > 1:
                break
            if self._j_seq[j - 1][0] != 'T':
                to_subs = f"T{cur_seq[1]}"
                jl_sequences[j - 1] = re.sub(to_subs, '', prev_seq)

        # Move the last link parameters to the tool
        jl_sequences[-1] = jl_sequences[-1][:2]

        i = 1
        result_sequence = ''
        variables, nominal_and_qs, joint_indices = [], [], []
        for jl_seq, p_seq, var in zip(jl_sequences, self._j_seq, self._j_var):
            # Get joint variable
            joint_indices.append(1)
            jl_tok = st._tokens_from_sequence(jl_seq[2:])
            p_tok = st._tokens_from_sequence(p_seq[2:])

            nominal_and_qs.append(var[0])
            if i == 1 or i == len(jl_sequences):
                joint_indices.extend([0] * len(jl_tok))
                result_sequence += jl_seq[:2]
                variables.append(var[0])
            else:
                joint_indices.extend([0] * (len(jl_seq) // 2))
                result_sequence += jl_seq[:2] + jl_seq[:2]
                variables.extend([var[0], f"d_{var[0]}"])
                nominal_and_qs.append(0)

            var = var[1:]
            for tok in jl_tok:
                result_sequence += tok
                tf = 'phi' if tok[0] == 'R' else 'p'
                variables.append(f"{tf}_{tok[1]}_{i}")

                if tok in p_tok:
                    nominal_and_qs.append(var[p_tok.index(tok)])
                else:
                    nominal_and_qs.append(0)
            i += 1

        parameters = [v for v, i in zip(variables, joint_indices) if i != 1]
        nominal = [v for v, i in zip(nominal_and_qs, joint_indices) if i != 1]
        parameter_indices = 1 - np.array(joint_indices)

        self._r_seq = result_sequence
        self._r_var = variables
        self._r_par = parameters
        self._parameters = np.zeros(len(parameters))
        self._r_nom_par = nominal
        self._r_nom_and_q = nominal_and_qs
        self._r_j_ind = joint_indices
        self._r_par_ind = parameter_indices
        self._n_params = sum(parameter_indices)

        self._jc = JacobianCalculator(self._r_seq,
                                      self._r_par_ind,
                                      f_of_t=False,
                                      variables=self._r_var)

        print("Went to base: ", end='')
        for tf, v in zip(self._to_base, self._to_base_vars):
            print(f"{tf}({v})", end='')
        print()
        print("Went to tool: ", end='')
        for tf, v in zip(self._to_tool, self._to_tool_vars):
            print(f"{tf}({v})", end=' ')
        print()

        return result_sequence, variables

    def estimate_tool_base(self, qs, pts, parameters=None):
        """
        Estimates Tbase and Ttools transformations from measurements

        Args:
            qs (np.ndarray): (m, k) array of measured joint angles
                m - number of samples measured
                k - number of joints manipulator has
            pts (np.ndarray): (m, 3n, 1) array of measured point coordinates
                m - number of samples measured
                n - number of measurement points
            Measrured points should be stacked like this:
                [x1 y1 z1, x2, y2, z2, ..., xn, yn, zn]

        Returns:
            sp.Matrix, list of sp.Matrix: Estimated homogeneous Tbase,
                list of n estimated homogeneous Ttools
        """
        if self._parameters is None:
            print(self._no_model_msg)
            return [], []

        self._base_iteration += 1

        # Just for sanity check. Assuming distances of points we measure
        # will not change, we can check if model and offsets are correct
        pairwise_1 = pairwise_2 = pairwise_3 = 0

        n = pts.shape[1] // 3
        dim = 6 + 3 * n

        Adp_sum = sp.zeros(dim, 1)
        A_sum = sp.zeros(dim, dim)
        desc = "Estimating Tbase and Ttools"
        for q, p in tqdm(zip(qs, pts), total=qs.shape[0], desc=desc):
            T_robot = self.get_pose(q)
            R_robot, p_robot = T_robot[:3, :3], T_robot[:3, 3]

            dp_i = sp.Matrix(p - np.tile(p_robot.T, n).T)

            pairwise_1 += np.linalg.norm(p[:3] - p[3:6])
            pairwise_2 += np.linalg.norm(p[6:] - p[3:6])
            pairwise_3 += np.linalg.norm(p[:3] - p[6:])

            p_scew = tf.get_scew(p_robot[0], p_robot[1], p_robot[2]).T
            A_i = sp.Matrix(0, dim, [])
            for i in range(n):
                if i == 0:
                    z = sp.zeros(3, (n - 1) * 3)
                    row = sp.eye(3).row_join(
                        p_scew).row_join(R_robot).row_join(z)
                if i == (n - 1):
                    z = sp.zeros(3, (n - 1) * 3)
                    row = sp.eye(3).row_join(
                        p_scew).row_join(z).row_join(R_robot)
                else:
                    z1 = sp.zeros(3, 3 * i)
                    z2 = sp.zeros(3, 3 * (n - 1 - i))
                    row = sp.eye(3).row_join(
                        p_scew).row_join(z1).row_join(R_robot).row_join(z2)

                A_i = A_i.col_join(row)

            A_sum += A_i.T * A_i
            Adp_sum += A_i.T * dp_i

            # sp.pprint(A_sum)
            # sp.pprint(Adp_sum)

        pairwise_1 /= qs.shape[0]
        pairwise_2 /= qs.shape[0]
        pairwise_3 /= qs.shape[0]

        # Sympy, propably, has some bug or overflows, so the order
        # of addition on the previous step influences invertability
        # So, converting to numpy
        A_sum = np.array(A_sum, dtype=np.float)
        Adp_sum = np.array(Adp_sum, dtype=np.float)

        estimated = np.linalg.inv(A_sum).dot(Adp_sum)[:, 0]

        s = self._step
        if self._base_iteration == 1:
            self._base_est = estimated
        else:
            self._base_est = (1 - s) * self._base_est + s * estimated

        est = self._base_est

        T_base = tf.get_Tx(est[0]) * tf.get_Ty(est[1]) * tf.get_Tz(est[2])
        T_base *= tf.get_Rx(est[3]) * tf.get_Ry(est[4]) * tf.get_Rz(est[5])

        T_tools = []

        for i in range(6, len(est), 3):
            T_tool_i = tf.get_Tx(est[i]) * tf.get_Ty(est[i + 1])
            T_tool_i *= tf.get_Tz(est[i + 2])
            T_tools.append(T_tool_i)

        # print("Pairwise distances real:")
        # print(pairwise_1)
        # print(pairwise_2)
        # print(pairwise_3)
        # print()

        # print("Pairwise distances estimated:")
        # print((T_tools[1][:3, 3] - T_tools[0][:3, 3]).norm())
        # print((T_tools[2][:3, 3] - T_tools[1][:3, 3]).norm())
        # print((T_tools[0][:3, 3] - T_tools[2][:3, 3]).norm())
        # print()

        return T_base, T_tools

    def estimate_parameters(self, qs, pts, T_base=None, T_tools=None):
        """
        Estimates geometric parameters from measurements

        Args:
            qs (np.ndarray): (m, k) array of measured joint angles
                m - number of samples measured
                k - number of joints manipulator has
            pts (np.ndarray): (m, 3n, 1) array of measured point coordinates
                m - number of samples measured
                n - number of measurement points
            Measrured points should be stacked like this:
                [x1 y1 z1, x2, y2, z2, ..., xn, yn, zn]
            T_base (4x4 sp.Matrix): T_base transformation
            T_tools (list of 4x4 sp.Matrix): List of T_tool transformations
                for each point of n points

        Returns:
            np.ndarray: Array with values of estimated parameters
        """
        if self._parameters is None:
            print(self._no_model_msg)
            return [], []

        if T_base is None:
            T_base = self._T_base

        if T_tools is None:
            T_tools = self._T_tools

        self._params_iteration += 1

        Jdp_sum = sp.zeros(self._n_params, 1)
        J_sum = sp.zeros(self._n_params, self._n_params)
        desc = "Estimating parameters"
        for q, p in tqdm(zip(qs, pts), total=qs.shape[0], desc=desc):
            for j, T_tool in enumerate(T_tools):
                T_robot = self.get_pose(q, T_base, T_tool)
                J = self.get_jacobian(q, T_base, T_tool)
                dp_j = p[j * 3:j * 3 + 3] - T_robot[:3, 3]
                J_sum += J.T * J
                Jdp_sum += J.T * dp_j

        # Sympy, propably, has some bug or overflows, so the order
        # of addition on the previous step influences invertability
        # So, converting to numpy
        J_sum = np.array(J_sum, dtype=np.float)
        Jdp_sum = np.array(Jdp_sum, dtype=np.float)

        parameters = np.linalg.inv(J_sum).dot(Jdp_sum)[:, 0]
        simga = 0.043883037319796446
        Cov = simga**2 * J_sum

        # to_apply = []
        # for i, parameter in enumerate(parameters):
        #     val = 0.0
        #     if 2.0 > np.sqrt(Cov[i, i]):
        #         val = parameter
        #     to_apply.append(val)
        # to_apply = np.array(to_apply)

        s = self._step
        if self._params_iteration == 1:
            self._parameters = parameters
        else:
            self._parameters = (1.0 - s) * self._parameters + s * parameters

        print()
        print("Parameters: ")
        for i, (var, val) in enumerate(zip(self._r_par, self._parameters)):
            print(f"{var} = {val:.4f} (+-{np.sqrt(Cov[i, i]):.4f})")
        print()

        return parameters

    def evaluate(self, qs, pts, T_base, T_tools):
        """
        Evaluates results of calibration

        Args:
            qs (np.ndarray): (m, k) array of measured joint angles
                m - number of samples measured
                k - number of joints manipulator has
            pts (np.ndarray): (m, 3n, 1) array of measured point coordinates
                m - number of samples measured
                n - number of measurement points
            Measrured points should be stacked like this:
                [x1 y1 z1, x2, y2, z2, ..., xn, yn, zn]
            T_base (4x4 sp.Matrix): T_base transformation
            T_tools (list of 4x4 sp.Matrix): List of T_tool transformations
                for each point of n points

        Returns:
            dict of str: float: Dictionary of {"metric name": result}
        """
        if self._parameters is None:
            print(self._no_model_msg)
            return [], []

        rms_dist = rms_x = rms_y = rms_z = 0.0
        max_dist = max_diff_x = max_diff_y = max_diff_z = -1.0
        for q, p in tqdm(zip(qs, pts), total=qs.shape[0], desc='Evaluating'):
            for j, T_tool in enumerate(T_tools):
                T_robot = self.get_pose(q, T_base, T_tool)
                diff = p[j * 3:j * 3 + 3] - T_robot[:3, 3]
                dist = diff.norm()
                diff_x = abs(diff[0])
                diff_y = abs(diff[1])
                diff_z = abs(diff[2])
                rms_dist += dist * dist
                rms_x += diff_x * diff_x
                rms_y += diff_y * diff_y
                rms_z += diff_z * diff_z

                if dist > max_dist:
                    max_dist = dist

                if diff_x > max_diff_x:
                    max_diff_x = diff_x

                if diff_y > max_diff_y:
                    max_diff_y = diff_y

                if diff_z > max_diff_z:
                    max_diff_z = diff_z

        n = pts.shape[0] * pts.shape[1] // 3
        rms_dist = sp.sqrt(rms_dist / n)
        rms_x = sp.sqrt(rms_x / n)
        rms_y = sp.sqrt(rms_y / n)
        rms_z = sp.sqrt(rms_z / n)

        print()
        print(f"Distance RMS (mm): {rms_dist}")
        print(f"Max Distance diff (mm): {max_dist}")
        print()
        print(f"X coordinate RMS (mm): {rms_x}")
        print(f"Max X diff (mm): {max_diff_x}")
        print()
        print(f"Y coordinate RMS (mm): {rms_y}")
        print(f"Max Y diff (mm): {max_diff_y}")
        print()
        print(f"Z coordinate RMS (mm): {rms_z}")
        print(f"Max Z diff (mm): {max_diff_z}")
        print()
        print()

        result = {
            "distance_rms": rms_dist,
            "x_rms": rms_x,
            "y_rms": rms_y,
            "z_rms": rms_z,
            "max_dist": max_dist,
            "max_diff_x": max_diff_x,
            "max_diff_y": max_diff_y,
            "max_diff_z": max_diff_z,
        }

        return result

    def get_pose(self, qs, T_base=None, T_tool=None, parameters=None):
        if T_base is None:
            T_base = sp.eye(4)
        if T_tool is None:
            T_tool = sp.eye(4)
        values = self._prepare_evaluation(qs, parameters)
        T_robot = T_base * tf(self._r_seq, values).transformation * T_tool
        return T_robot

    def get_jacobian(self, qs, T_base, T_tool):
        values = self._prepare_evaluation(qs)
        Jp = self._jc.calculate_numerically(values, T_base, T_tool)[:3, :]
        return Jp

    def _prepare_evaluation(self, qs, parameters=None):
        # Apply angle offsets and rotation directions
        qs = np.multiply(qs, self.qs_directions) + self.qs_offsets

        if parameters is None:
            parameters = self._parameters

        val_dict = {}
        i = 0
        for name, is_joint in zip(self._r_var, self._r_j_ind):
            if is_joint:
                val_dict[name] = qs[i]
                i += 1
        for pair in self._link_lengths:
            val_dict[pair[0]] = pair[1]

        i = 0
        res = []
        for name, is_joint in zip(self._r_var, self._r_j_ind):
            if is_joint:
                res.append(val_dict[name])
                continue
            nom_val = 0.0
            if self._r_nom_par[i] in val_dict:
                nom_val = val_dict[self._r_nom_par[i]]
            res.append(nom_val + parameters[i])
            i += 1

        return res

    def optimize(self, qs, pts, T_base, T_tools):
        kwargs = {"T_base": T_base, "T_tools": T_tools, "qs": qs, "pts": pts}
        p_0 = self._parameters

        def cost(parameters, qs, pts, T_base, T_tools):
            error = 0.0
            for q, p in zip(qs, pts):
                for j, T_tool in enumerate(T_tools):
                    T_robot = self.get_pose(q, T_base, T_tool, parameters)
                    diff = p[j * 3:j * 3 + 3] - T_robot[:3, 3]
                    res = float(diff.norm())
                    # Squared res optimizes MSE
                    # Just res optimizes max deviation
                    error += res * res
            return error

        res = least_squares(cost, p_0, kwargs=kwargs,
                            jac='3-point',
                            ftol=1e-4,
                            gtol=1e-4,
                            xtol=None,
                            verbose=2)

        self._parameters = res.x

    def optimize_with_base(self, qs, pts):
        kwargs = {"qs": qs, "pts": pts}
        p_0 = np.concatenate((self._parameters, self._base_est))

        def cost(x, qs, pts):
            n = len(self._parameters)
            parameters = x[:n]
            est = x[n:]

            T_base = tf.get_Tx(est[0]) * tf.get_Ty(est[1]) * tf.get_Tz(est[2])
            T_base *= tf.get_Rx(est[3]) * tf.get_Ry(est[4]) * tf.get_Rz(est[5])

            T_tools = []

            for i in range(6, len(est), 3):
                T_tool_i = tf.get_Tx(est[i]) * tf.get_Ty(est[i + 1])
                T_tool_i *= tf.get_Tz(est[i + 2])
                T_tools.append(T_tool_i)

            error = 0.0
            for q, p in zip(qs, pts):
                for j, T_tool in enumerate(T_tools):
                    T_robot = self.get_pose(q, T_base, T_tool, parameters)
                    diff = p[j * 3:j * 3 + 3] - T_robot[:3, 3]
                    res = float(diff.norm())
                    # res * res  optimizes MSE
                    # Just res optimizes max deviation
                    error += res * res
            return error

        res = least_squares(cost, p_0, kwargs=kwargs,
                            jac='3-point',
                            ftol=1e-3,
                            gtol=1e-3,
                            xtol=1e-3,
                            verbose=2)

        print(res)

        self._parameters = res.x
