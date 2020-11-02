import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class TransformationPlotter():
    ax = plt.axes(projection='3d')

    @staticmethod
    def plot(T, values, to_plot_frame=None):
        if to_plot_frame is None:
            to_plot_frame = {v for v in T.variables}

        values_dict = {}
        for value, name in zip(values, T.variables):
            values_dict[sp.symbols(name)] = value

        joint_points = []

        TransformationPlotter._plot_frame(np.eye(4))
        joint_points.append(np.array([[0], [0], [0]]))

        for frame, variable in zip(T.frames, T.variables):
            numeric_T = frame.evalf(subs=values_dict)
            joint_points.append(numeric_T[:3, 3])

            if variable in to_plot_frame:
                TransformationPlotter._plot_frame(numeric_T)

        TransformationPlotter._plot_frame(numeric_T)

        joint_points = np.array(joint_points)

        TransformationPlotter._plot_links(joint_points)
        TransformationPlotter.show()

    @staticmethod
    def plot_numeric_frames(frames):
        joint_points = []

        TransformationPlotter._plot_frame(np.eye(4), joint_color='orange')
        joint_points.append(np.array([[0], [0], [0]]))

        for frame in frames:
            joint_points.append(frame[:3, 3])
            TransformationPlotter._plot_frame(np.array(frame))

        joint_points = np.array(joint_points, dtype=float)

        max_range = np.array([
            joint_points[:, 0, 0].max() - joint_points[:, 0, 0].min(),
            joint_points[:, 1, 0].max() - joint_points[:, 1, 0].min(),
            joint_points[:, 2, 0].max() - joint_points[:, 2, 0].min()
        ]).max() / 2.0

        mid_x = (joint_points[:, 0, 0].max() + joint_points[:, 0, 0].min()) / 2
        mid_y = (joint_points[:, 1, 0].max() + joint_points[:, 1, 0].min()) / 2
        mid_z = (joint_points[:, 2, 0].max() + joint_points[:, 2, 0].min()) / 2

        TransformationPlotter.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        TransformationPlotter.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        TransformationPlotter.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        TransformationPlotter._plot_links(joint_points)
        TransformationPlotter.show()

    @staticmethod
    def _plot_frame(T,
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

        TransformationPlotter.ax.plot3D(
            x[:, 0], x[:, 1], x[:, 2], 'r-', linewidth=line_width)
        TransformationPlotter.ax.plot3D(
            y[:, 0], y[:, 1], y[:, 2], 'g-', linewidth=line_width)
        TransformationPlotter.ax.plot3D(
            z[:, 0], z[:, 1], z[:, 2], 'b-', linewidth=line_width)

        origin = vectors[:, 0]
        TransformationPlotter.ax.scatter(
            origin[0],
            origin[1],
            origin[2],
            c=joint_color,
            s=joint_size,
        )

    @staticmethod
    def _plot_links(joint_points,
                    link_color='black',
                    link_width=3):
        TransformationPlotter.ax.plot3D(
            np.ndarray.flatten(joint_points[:, 0]),
            np.ndarray.flatten(joint_points[:, 1]),
            np.ndarray.flatten(joint_points[:, 2]),
            linewidth=link_width,
            c=link_color,
            alpha=0.4
        )

    @staticmethod
    def show():
        plt.tight_layout()
        plt.show()
        plt.cla()
