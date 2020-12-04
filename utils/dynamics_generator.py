"""
DynamicsGenerator class definition
"""
import numpy as np
import sympy as sp
from tqdm import tqdm

from utils.robo_math import SymbolicTransformation as st
from utils.jacobians import JacobianCalculator

from utils.plot_utils import LatexRenderer
from utils.plot_utils import TrajectoriesPlotter
from utils.constants import G_ACC


class DynamicsGenerator():
    """
    Solves forward and inverse dynamics problems

    Attributes:
        simplify (bool, optional): Flag to apply symbolic simplification
        T_base (None, optional): Transformation from the world frame
            to the base frame
        T_tool (None, optional): Transformation from the end-effector
            frame to the tool frame
    """

    def __init__(self,
                 sequence_string,
                 mass_center_indices,
                 joint_indices,
                 variables=None,
                 T_base=None,
                 T_tool=None,
                 gravity_axis="z",
                 print_equation=False,
                 simplify=False,
                 method='lagrange',
                 no_motor=True):
        """
        Initialization and equation precalculation

        Args:
            sequence_string (str): Transformation sequence
            mass_center_indices (list of bool): Mask list with True on elements
                that are transformations for centers of masses
            joint_indices (list of bool): Mask list with True on elements with
                that are not constant
            variables (None, optional): List with names of variables
            T_base (None, optional): Transformation from the world frame
                to the base frame
            T_tool (None, optional): Transformation from the end-effector
                frame to the tool frame
            gravity_axis (str, optional): Name of the axis of gravity: x, y, z
            print_equation (bool, optional): Flag to print the obtained
                differential equation
            simplify (bool, optional): Flag to apply symbolic simplification
            method (str, optional): Method to apply. Newton-Euler or Lagrange
            no_motor (bool, optional): Flag to consider motors in calculation

        Raises:
            ValueError: Error when dimensions do not match or incorrect
                sequence string is provided
        """
        self._seq = sequence_string
        self._tokens = st._tokens_from_sequence(self._seq)
        self._cm_ind = mass_center_indices
        self._joint_ind = joint_indices
        self._lg_seqs = []
        self._lg_vars = []
        self._lg_j_inds = []
        self._ne_seqs = []
        self._ne_vars = []
        self._ne_j_inds = []
        self._ne_cm_inds = []
        self._validator = st('')
        self._gravity_axis = ord(gravity_axis.lower()) - ord('x')
        self._g_const = G_ACC
        self._print_equation = print_equation
        self.simplify = simplify
        self._no_motor = no_motor
        self._method = method.lower()

        self._lagrange_names = {'lagrange', 'l'}
        self._newton_euler_names = {'newton', 'euler', 'ne', 'newton-euler'}

        if self._method in self._lagrange_names:
            self._calculate_equations = self._calculate_equations_lagrange
        elif self._method in self._newton_euler_names:
            self._calculate_equations = self._calculate_equations_newton_euler
        else:
            print("[INFO] Unknown method specified, defaulting to lagrange")
            self._calculate_equations = self._calculate_equations_lagrange

        self.set_transforms(T_base, T_tool)

        if len(self._tokens) != len(self._cm_ind):
            raise ValueError("Size of mass center indices does not match")

        if len(self._tokens) != len(self._joint_ind):
            raise ValueError("Size of joint indices does not match")

        # Generate variable names if necessary
        if variables is None:
            self._variables = []
            t_index = 0
            r_index = 0

            for token, index in zip(self._tokens, self._cm_ind):
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

        # Generate sequences of transforms for the lagrange method
        for i, (is_cm) in enumerate(self._cm_ind, 1):
            if is_cm:
                self._lg_seqs.append(''.join(self._tokens[:i]))
                self._lg_vars.append(self._variables[:i])
                self._lg_j_inds.append(self._joint_ind[:i])

        # Generate sequences of transforms for newton-euler method
        indices = [i for i, x in enumerate(self._joint_ind + [1]) if x == 1]
        for i, j in zip(indices, indices[1:]):
            self._ne_seqs.append(''.join(self._tokens[i:j]))
            self._ne_vars.append(self._variables[i:j])
            self._ne_j_inds.append(self._joint_ind[i:j])
            self._ne_cm_inds.append(self._cm_ind[i:j])

        self._calculate_equations()

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

    def _calculate_equations_lagrange(self):
        """
        Calculates differential equations using Euler-Lagrange method
        """
        index = 0
        g, t = sp.Matrix([0, 0, 0]), sp.Symbol("t")
        g[self._gravity_axis] = sp.Symbol("g")
        K, P = sp.Matrix([[0]]), sp.Matrix([[0]])
        for sq, var, ind in zip(self._lg_seqs, self._lg_vars, self._lg_j_inds):
            J_i = JacobianCalculator(sequence_string=sq,
                                     joint_indices=ind,
                                     variables=var,
                                     simplify=True).calculate_scew()
            # TODO, check T_base transform correctness
            T_i = self.T_base * st(sq, var, f_of_t=ind).transformation
            p_i = T_i[:3, 3]

            m_i = sp.symbols(f"m_{index}")
            I_i = st.get_inertia_matrix(index)

            dqs = []
            for v, is_joint in zip(var, ind):
                if is_joint:
                    dqs.append(sp.Function(v)(t).diff(t))

            twist_i = J_i * sp.Matrix(dqs)
            v_i = twist_i[:3, :]
            w_i = twist_i[3:, :]

            K += (m_i * v_i.T * v_i + w_i.T * I_i * w_i)
            P += m_i * g.T * p_i

            index += 1

        L = 0.5 * K - P

        qs, dqs, equations = [], [], []
        for v, is_joint in zip(self._lg_vars[-1], self._lg_j_inds[-1]):
            if is_joint:
                q = sp.Function(v)(t)
                dq = sp.Function(v)(t).diff(t)
                eq = L.diff(dq).diff(t) - L.diff(q)

                qs.append(q)
                dqs.append(dq)
                equations.append(sp.simplify(eq))
        self._equations = equations

        if self._print_equation:
            LatexRenderer.render_equations(self._equations, qs + dqs)

    def _calculate_equations_newton_euler(self):
        """
        Calculates differential equations using Newton-Euler method
        """
        g = sp.Matrix([0, 0, 0])
        g[self._gravity_axis] = -sp.Symbol("g")

        # TODO: pass them as parameters
        w_0 = sp.Matrix([0, 0, 0])
        dw_0 = sp.Matrix([0, 0, 0])
        ddp_0 = sp.Matrix([0, 0, 0])
        f_n = sp.Matrix([0, 0, 0])
        mu_n = sp.Matrix([0, 0, 0])

        dqs, ddqs = [], []
        for v, is_joint in zip(self._variables, self._joint_ind):
            if is_joint:
                q_funct = sp.Function(v)("t")
                ddqs.append(sp.Derivative(q_funct, ("t", 2)))
                dqs.append(sp.Derivative(q_funct, "t"))
        dqs = sp.Matrix(dqs)
        ddqs = sp.Matrix(ddqs)

        Rs, rs, rcs = [], [sp.Matrix([0, 0, 0])], []
        ws, dws, ddps, joint_types = [w_0], [dw_0], [ddp_0 - g], []
        zs, ddpcs, dwms = [], [], []
        for i in range(1, len(self._ne_seqs) + 1):
            z_i = sp.Matrix([0, 0, 0])
            z_i[ord(self._ne_seqs[i - 1][1]) - ord('x')] = 1

            R_i = st(self._ne_seqs[i - 1], self._ne_vars[i - 1],
                     self._ne_j_inds[i - 1])[:3, :3]

            r_ind = 0
            if (self._ne_seqs[i - 1][0] == "R"):
                r_ind = 1
            r_seq = self._ne_seqs[i - 1][r_ind * 2:]
            r_vars = self._ne_vars[i - 1][r_ind:]

            rc_i = sp.Matrix([0, 0, 0])
            if 1 in self._ne_cm_inds[i - 1][r_ind:]:
                cm_ind = self._ne_cm_inds[i - 1][r_ind:].index(1)
                rc_i_seq = r_seq[:(cm_ind + 1) * 2]
                rc_i_vars = r_vars[:cm_ind + 1]
                rc_i = st(rc_i_seq, rc_i_vars)[:3, 3]
            r_i = st(r_seq, r_vars)[:3, 3]
            r_i_ci = rc_i - r_i

            Rs.append(R_i)
            rs.append(r_i)
            rcs.append(rc_i)

            # Calculation of w, dw, ddp
            if self._ne_seqs[i - 1][0] == "T":
                joint_types.append("T")

                w_i = R_i.T * ws[i - 1]
                dw_i = R_i.T * dws[i - 1]

                tmp = 2 * dqs[i - 1] * (w_i.cross(R_i.T * z_i))
                tmp_1 = dw_i.cross(r_i) + w_i.cross(w_i.cross(r_i))
                ddp_i = R_i.T * (ddps[i - 1] + ddqs[i - 1] * z_i) + tmp + tmp_1
            else:
                joint_types.append("R")

                w_i = R_i.T * (ws[i - 1] + dqs[i - 1] * z_i)

                tmp = (dqs[i - 1] * ws[i - 1]).cross(z_i)
                dw_i = R_i.T * (dws[i - 1] + ddqs[i - 1] * z_i + tmp)

                tmp = w_i.cross(w_i.cross(r_i))
                ddp_i = R_i.T * ddps[i - 1] + dw_i.cross(r_i) + tmp

            ws.append(w_i)
            dws.append(dw_i)
            ddps.append(ddp_i)

            # Calculation of ddpc, dwm
            ddpc_i = ddp_i + dw_i.cross(r_i_ci) + w_i.cross(w_i.cross(r_i_ci))
            kr_i = sp.Symbol(f"kr_{i}")
            if self._no_motor:
                kr_i = 1.0

            tmp = kr_i * dqs[i - 1] * (ws[i - 1].cross(z_i))
            dwm_i = dws[i - 1] + kr_i * ddqs[i - 1] * z_i + tmp

            zs.append(z_i)
            ddpcs.append(ddpc_i)
            dwms.append(dwm_i)

        Rs.append(self.T_tool[:3, :3])

        # Backward pass
        zeros = [0] * (len(dqs))
        fs, mus, us = zeros + [f_n], zeros + [mu_n], zeros.copy()

        for i in range(len(fs) - 2, -1, -1):
            m_i = sp.Symbol(f"m_{i}")
            kr_i, kr_i1 = sp.Symbol(f"kr_{i}"), sp.Symbol(f"kr_{i+1}")
            Im_i, Im_i1 = sp.Symbol(f"Im_{i}"), sp.Symbol(f"Im_{i+1}")
            if self._no_motor:
                kr_i1 = kr_i1 = 1.0
                Im_i = Im_i1 = 0.0
            I_i = st.get_inertia_matrix(i)

            # Calculate current joint force
            f_i = Rs[i + 1] * fs[i + 1] + m_i * ddpcs[i]

            # Calculate current joint torque
            t = (Rs[i + 1] * fs[i + 1]).cross(rcs[i] - rs[i + 1])
            t1 = I_i * dws[i + 1] + ws[i + 1].cross(I_i * ws[i + 1])
            m = kr_i1 * ddqs[i] * Im_i1 * zs[i]
            m1 = (kr_i1 * dqs[i] * Im_i1 * ws[i]).cross(zs[i])
            mu_i = Rs[i + 1] * mus[i + 1] - f_i.cross(rcs[i]) + t + t1 + m + m1

            # Calculate current joint input
            temp = kr_i * Im_i * dwms[i - 1].T * zs[i]
            if joint_types[i] == "T":
                u_i = f_i.T * zs[i] + temp
            else:
                u_i = mu_i.T * zs[i] + temp

            fs[i], mus[i], us[i] = f_i, mu_i, u_i

        self._equations = []
        for u in us:
            self._equations.append(sp.simplify(u))

        if self._print_equation:
            LatexRenderer.render_equations(self._equations, dqs)

    def generate_control(self, q_func, dq_func, ddq_func,
                         parameters, T, dt=0.01, plot=False):
        """
        Generates control values for the desired trajectory

        Args:
            q_func (funciton): Function of time that returns desired q(t)
            dq_func (funciton): Function of time that returns desired dq(t)
            ddq_func (funciton): Function of time that returns desired ddq(t)
            parameters (list of (str, float)): Parameters to substitute in the
                symbolic equation
            T (float): Simulation time
            dt (float, optional): Simulation timestep
            plot (bool, optional): Flag to plot the result

        Returns:
            np.array: Controls to generate the desired trajectory
        """
        simplified_eqs, acc_vars, j_var = self._apply_parameters(parameters)

        print("Generating controls")
        ts = np.linspace(0, T, int(T / dt))
        us = []
        for t in tqdm(ts):
            cur_eqs = simplified_eqs.copy()
            cur_q = q_func(t)
            cur_dq = dq_func(t)
            cur_ddq = ddq_func(t)

            cur_eqs = self._apply_qs(cur_eqs, cur_q, cur_dq, cur_ddq, j_var)

            us.append(cur_eqs)

        us = np.array(us, dtype=np.float)[:, :, 0, 0]

        if plot:
            TrajectoriesPlotter.plot_control(ts, us.T)

        return us

    def generate_trajectory(self,
                            q_0,
                            dq_0,
                            u_func,
                            parameters,
                            T,
                            dt=0.01,
                            plot=False):
        """
        Generates trajectory from controls and initial state by numerically
        solving differential equations

        Args:
            q_0 (list of float): Initial qs
            dq_0 (list of float): Initial dqs
            u_func (function): Function of time that returns contols at
                time t
            parameters (list of (str, float)): Parameters to substitute in the
                symbolic equation
            T (float): Simulation time
            dt (float, optional): Simulation timestep
            plot (bool, optional): Flag to plot the result

        Returns:
            np.ndarray: Simulated qs
        """
        simplified_eqs, acc_vars, j_var = self._apply_parameters(parameters)

        print("Integrating equations")
        ts = np.linspace(0, T, int(T / dt))
        qs, dqs, ddqs, us = [sp.Matrix(q_0)], [sp.Matrix(dq_0)], [], []
        for t in tqdm(ts):
            cur_eqs = simplified_eqs.copy()
            cur_u = u_func(t)
            cur_q, cur_dq = qs[-1], dqs[-1]

            cur_eqs = self._apply_qs(cur_eqs, cur_q, cur_dq, acc_vars, j_var)

            to_solve = []
            for eq, u in zip(cur_eqs, cur_u):
                to_solve.append(eq[0] - u)

            # TODO introduce better integraion technique
            cur_ddq = sp.Matrix(list(sp.solve(to_solve, acc_vars).values()))
            new_dq = cur_dq + cur_ddq * dt
            new_q = cur_q + cur_dq * dt

            if np.linalg.norm(np.array(cur_ddq, dtype=float)) > 100:
                print("[ERROR] |ddq| > 100")
                break

            us.append(cur_u)
            ddqs.append(cur_ddq)
            dqs.append(new_dq)
            qs.append(new_q)

        us = np.array(us)
        qs = np.array(qs, dtype=np.float)[:-1, :, 0]
        dqs = np.array(dqs, dtype=np.float)[:-1, :, 0]
        ddqs = np.array(ddqs, dtype=np.float)[:, :, 0]
        ts = ts[:len(qs)]

        if plot:
            TrajectoriesPlotter.plot_joint(ts, qs.T, dqs.T, ddqs.T)

        return qs

    def _apply_parameters(self, parameters):
        """
        Substitutes list of parameters into symbolic equations

        Args:
            parameters (list of (str, float)): Parameters to substitute

        Returns:
            sp.expression, list of sp.symbol, list of str:
                Result of substitution,
                List of variables to solve equations for,
                List of symbolic names of joints
        """
        j_var = [v for v, i in zip(self._variables, self._joint_ind) if i == 1]
        acc_vars = []

        if len([item for item in parameters if 'g' in item]) == 0:
            parameters.append(('g', self._g_const))

        simplified_eqs = self._equations.copy()
        for i in range(len(simplified_eqs)):
            simplified_eqs[i] = simplified_eqs[i].subs(parameters)
            acc_vars.append(sp.Symbol(f"a_{i}"))

        return simplified_eqs, acc_vars, j_var

    def _apply_qs(self, equations, qs, dqs, ddqs, joint_variables):
        """
        Substitutes qs, dqs, ddqs with provided values

        Args:
            equations (sp.expression): Equations to substitute in
            qs (list of sp.rvalue): Values to substitute qs with
            dqs (list of sp.rvalue): Values to substitute dqs with
            ddqs (list of sp.rvalue): Values to substitute ddqs with
            joint_variables (list of str): List with symbolic names of
                joints

        Returns:
            sp.expression: Result of substitution
        """
        for q, dq, ddq, joint_variable in zip(qs, dqs, ddqs, joint_variables):
            num_q = [(sp.Symbol(joint_variable), q)]
            q_funct = sp.Function(joint_variable)("t")
            sym_ddq = [(sp.Derivative(q_funct, ("t", 2)), ddq)]
            sym_dq = [(sp.Derivative(q_funct, "t"), dq)]
            sym_q = [(q_funct, q)]
            # Important to substitute in this order
            # q, dq, ddq are dependent
            for i in range(len(equations)):
                equations[i] = equations[i].subs(sym_ddq)
                equations[i] = equations[i].subs(sym_dq)
                equations[i] = equations[i].subs(sym_q)
                equations[i] = equations[i].subs(num_q)

        return equations
