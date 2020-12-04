import re

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import rc


class LatexRenderer():
    @staticmethod
    def render_equations(equations, variables):
        rc('text', usetex=True)
        latex_render = r'\begin{eqnarray*} '
        for i, eq in enumerate(equations):
            eq = sp.collect(eq[0], variables)
            latex_text = re.sub(r'\.0|1.0| |\\left|\\right', '', sp.latex(eq))
            latex_text = re.sub(r'\{\(t\)\}', '', latex_text)
            latex_text = re.sub(
                r'\\frac\{d\}\{dt\}', r'\\dot', latex_text)
            latex_text = re.sub(
                r'\\frac\{d\^\{2\}\}\{dt\^\{2\}\}', r'\\ddot', latex_text)
            latex_text += f"=u_{i}(t) \\\\"
            latex_render += latex_text
        latex_render += r'\end{eqnarray*}'

        fig = plt.figure(figsize=(20, 7))
        fig.subplots_adjust(wspace=0.05)
        plt.text(0.05, 0.5, latex_render, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class TrajectoriesPlotter():
    def plot_joint(ts, qs, dqs, ddqs):
        rc('text', usetex=False)
        fig, axs = plt.subplots(len(qs), 3, sharex=True, figsize=(20, 10))
        fig.suptitle("Joint Trajectories")
        labels = ["", r"\dot", r"\ddot"]
        for i, (q, dq, ddq) in enumerate(zip(qs, dqs, ddqs)):
            plots = [q, dq, ddq]
            for j, (plot, label) in enumerate(zip(plots, labels)):
                if len(qs) != 1:
                    axs[i, j].plot(ts, plot)
                    axs[i, j].set_title(f"${label}q_{i}$")
                    axs[i, j].set_xlabel(f'$t (s)$')
                    axs[i, j].set_ylabel(f'$q (rad)$')
                else:
                    axs[j].plot(ts, plot)
                    axs[j].set_title(f"${label}q_{i}$")
                    axs[j].set_xlabel(f'$t (s)$')
                    axs[j].set_ylabel(f'$q (rad)$')
        plt.tight_layout()
        plt.show()

    def plot_cartesian(ts, qs, dqs, ddqs):
        rc('text', usetex=False)
        fig, axs = plt.subplots(len(qs), 3, sharex=True, figsize=(20, 10))
        fig.suptitle("Cartesian Trajectories")
        labels = ["", r"\dot", r"\ddot"]
        coords = ["x", "y", "z"]
        for i, (q, dq, ddq, crd) in enumerate(zip(qs, dqs, ddqs, coords)):
            plots = [q, dq, ddq]
            for j, (plot, label) in enumerate(zip(plots, labels)):
                if len(qs) != 1:
                    axs[i, j].plot(ts, plot)
                    axs[i, j].set_title(f"${label}{crd}$")
                    axs[i, j].set_xlabel(f'$t (s)$')
                    axs[i, j].set_ylabel(f'${crd} (m)$')
                else:
                    axs[j].plot(ts, plot)
                    axs[j].set_title(f"${label}{crd}$")
                    axs[j].set_xlabel(f'$t (s)$')
                    axs[j].set_ylabel(f'${crd} (m)$')
        plt.tight_layout()
        plt.show()

    def plot_joint_no_acc(ts, qs, dqs):
        rc('text', usetex=False)
        fig, axs = plt.subplots(len(qs), 2, sharex=True, figsize=(20, 10))
        fig.suptitle("Joint Trajectories")
        labels = ["", r"\dot"]
        for i, (q, dq) in enumerate(zip(qs, dqs)):
            plots = [q, dq]
            for j, (plot, label) in enumerate(zip(plots, labels)):
                if len(qs) != 1:
                    axs[i, j].plot(ts, plot)
                    axs[i, j].set_title(f"${label}q_{i}$")
                    axs[i, j].set_xlabel(f'$t (s)$')
                    axs[i, j].set_ylabel(f'$q (rad)$')
                else:
                    axs[j].plot(ts, plot)
                    axs[j].set_title(f"${label}q_{i}$")
                    axs[j].set_xlabel(f'$t (s)$')
                    axs[j].set_ylabel(f'$q (rad)$')
        plt.tight_layout()
        plt.show()

    def plot_control(ts, us):
        rc('text', usetex=False)
        fig, axs = plt.subplots(len(us), sharex=True, figsize=(20, 10))
        fig.suptitle("Joint Control")
        for i, u in enumerate(us):
            axs[i].plot(ts, u)
            axs[i].set_title(f"$u_{i}$")
            axs[i].set_xlabel(f'$t (s)$')
            axs[i].set_ylabel(f'$u_{i}$')
        plt.tight_layout()
        plt.show()


class TransformationPlotter():
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self, T, values, to_plot_frame=None, show=True):
        if to_plot_frame is None:
            to_plot_frame = {v for v in T.variables}

        values_dict = {}
        for value, name in zip(values, T.variables):
            values_dict[sp.symbols(name)] = value

        joint_points = []

        self._plot_frame(np.eye(4))
        joint_points.append(np.array([[0], [0], [0]]))

        for frame, variable in zip(T.frames, T.variables):
            numeric_T = frame.evalf(subs=values_dict)
            joint_points.append(numeric_T[:3, 3])

            if variable in to_plot_frame:
                self._plot_frame(numeric_T)

        self._plot_frame(numeric_T)

        joint_points = np.array(joint_points)

        self._plot_links(joint_points)
        if show:
            self.show()

    def plot_position(self, T, color='magenta', size=10, alpha=0.5, show=True):
        x, y, z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])

        self.ax.scatter(
            x,
            y,
            z,
            c=color,
            s=size,
            alpha=alpha,
        )
        if show:
            self.show()

    def plot_numeric_frames(self, frames, axis_len=100, margin=1,
                            center=0, fixed_scale=False, show=True):
        joint_points = []

        self._plot_frame(np.eye(4),
                         joint_color='orange',
                         axis_len=axis_len)
        joint_points.append(np.array([[0], [0], [0]]))

        for frame in frames:
            joint_points.append(frame[:3, 3])
            self._plot_frame(np.array(frame),
                             axis_len=axis_len)

        joint_points = np.array(joint_points, dtype=float)

        max_range = np.array([
            joint_points[:, 0, 0].max() - joint_points[:, 0, 0].min(),
            joint_points[:, 1, 0].max() - joint_points[:, 1, 0].min(),
            joint_points[:, 2, 0].max() - joint_points[:, 2, 0].min()
        ]).max() * margin

        mid_x = (joint_points[:, 0, 0].max() + joint_points[:, 0, 0].min()) / 2
        mid_y = (joint_points[:, 1, 0].max() + joint_points[:, 1, 0].min()) / 2
        mid_z = (joint_points[:, 2, 0].max() + joint_points[:, 2, 0].min()) / 2

        if fixed_scale:
            max_range = margin
            mid_x = mid_y = mid_z = center

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self._plot_links(joint_points)
        if show:
            self.show()

    def _plot_frame(self,
                    T,
                    joint_color='magenta',
                    joint_size=40,
                    axis_len=100,
                    line_width=2):

        vector_extractor = np.array([
            [0.0, axis_len, 0.0, 0.0],
            [0.0, 0.0, axis_len, 0.0],
            [0.0, 0.0, 0.0, axis_len],
            [1.0, 1.0, 1.0, 1.0]
        ])

        vectors = np.dot(T, vector_extractor)

        x = np.array([vectors[:3, 0], vectors[:3, 1]])
        y = np.array([vectors[:3, 0], vectors[:3, 2]])
        z = np.array([vectors[:3, 0], vectors[:3, 3]])

        self.ax.plot3D(
            x[:, 0], x[:, 1], x[:, 2], 'r-', linewidth=line_width)
        self.ax.plot3D(
            y[:, 0], y[:, 1], y[:, 2], 'g-', linewidth=line_width)
        self.ax.plot3D(
            z[:, 0], z[:, 1], z[:, 2], 'b-', linewidth=line_width)

        origin = vectors[:, 0]
        self.ax.scatter(
            origin[0],
            origin[1],
            origin[2],
            c=joint_color,
            s=joint_size,
        )

    def _plot_links(self,
                    joint_points,
                    link_color='black',
                    link_width=3):
        self.ax.plot3D(
            np.ndarray.flatten(joint_points[:, 0]),
            np.ndarray.flatten(joint_points[:, 1]),
            np.ndarray.flatten(joint_points[:, 2]),
            linewidth=link_width,
            c=link_color,
            alpha=0.4
        )

    def show(self):
        plt.tight_layout()
        plt.show()
