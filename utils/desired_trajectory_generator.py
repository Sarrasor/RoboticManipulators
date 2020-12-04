"""
Several sample trajectories
"""
import numpy as np
import matplotlib.pyplot as plt


def func_from_array(array, T):
    """
    Creates a time-dependent functio

    Args:
        array (TYPE): Description
        T (TYPE): Description

    Returns:
        function: Function of time
    """
    def func(t):
        index = int((len(array) - 1) * (t / T))
        if 0 <= index <= len(array) - 1:
            return array[index]
        else:
            return np.zeros_like(array)
    return func


def line_trajectory():
    """
    Generates line trajectory

    Returns:
        np.ndarray: Array of 3D points that are sorted left to right
    """
    dt = 0.01
    y = np.arange(-1, 1, dt)
    x = np.ones_like(y) * 0.5
    z = np.zeros_like(y)
    points = np.array([x, y, z]).T

    return points


def ellipse_trajectory():
    """
    Generates ellipse trajectory

    Returns:
        np.ndarray: Array of 3D points that are sorted left to right
    """
    dt = 0.01
    a = 0.5
    b = 1
    y_0 = 0
    x_0 = 0
    ts = np.arange(-np.pi, np.pi, dt)
    y = y_0 + a * np.sin(ts)
    x = x_0 + b * np.cos(ts)
    z = np.zeros_like(y)
    points = np.array([x, y, z]).T

    return points


def batman_trajectory():
    """
    Generates batman trajectory

    Returns:
        np.ndarray: Array of 3D points that are sorted left to right
    """
    step = 0.005
    x_result = np.zeros(0)
    y_result = np.zeros(0)

    # Bottom part
    xs = np.arange(-4, 4, step)
    temp = np.sqrt(1 - (np.abs(np.abs(xs) - 2) - 1)**2) - 3
    ys = abs(xs / 2) - 0.09137 * xs**2 + temp
    x_result = np.concatenate((x_result, xs[1:]))
    y_result = np.concatenate((y_result, ys[1:]))

    # Wings right lower
    xs = np.arange(4, 7, step * 1.5)
    ys = -3 * np.sqrt(-(xs / 7)**2 + 1)
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Wings right upper
    xs = np.arange(3, 7, step * 1.5)
    xs = np.flip(xs)
    ys = 3 * np.sqrt(-(xs / 7)**2 + 1)
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Head-wings connection right
    xs = np.arange(1, 2.92, step)
    xs = np.flip(xs)
    ys = 1.5 - .5 * abs(xs) - 1.89736 * (np.sqrt(3 - xs**2 + 2 * abs(xs)) - 2)
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Head right
    xs = np.arange(.8, 1, step / 2)
    xs = np.flip(xs)
    ys = 9 - 8 * abs(xs)
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Ears right
    xs = np.arange(.45, .7, step / 4)
    xs = np.flip(xs)
    ys = 3 * abs(xs) + .75
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Head top
    xs = np.arange(-.45, .45, step)
    xs = np.flip(xs)
    ys = np.ones_like(xs) * 2.1
    x_result = np.concatenate((x_result, xs))
    y_result = np.concatenate((y_result, ys))

    # Smooth the corners
    N = 31
    x_result = np.convolve(x_result, np.ones((N,)) / N)[(N - 1):]
    y_result = np.convolve(y_result, np.ones((N,)) / N)[(N - 1):]
    x_result = x_result[:-50]
    y_result = y_result[:-50]

    # Mirror
    idices = x_result > 0
    x_result = x_result[idices]
    y_result = y_result[idices]
    x_result = np.concatenate((x_result, -x_result[::-1]))
    y_result = np.concatenate((y_result, y_result[::-1]))
    shift = -(np.argmin(x_result) + 1)
    x_result = np.roll(x_result, shift)
    y_result = np.roll(y_result, shift)

    # Scale and translate
    x_offset = 0
    y_offset = 0
    scale = 0.22
    x_result = (scale * x_result) + x_offset
    y_result = (scale * y_result) + y_offset
    z_result = np.zeros_like(x_result)

    points = np.array([x_result, y_result, z_result]).T

    return points


def interpolate(p_start, p_finish, n):
    """
    Interpolates n values between two points

    Args:
        p_start (np.ndarray): Start point
        p_finish (np.ndarray): Finish point
        n (int): Nuber of points

    Returns:
        np.ndarray: Result of interpolation
    """
    v = np.array([p_finish]) - np.array([p_start])
    t = np.array([np.linspace(0, 1, n)]).T
    return p_start + t.dot(v)


def main():
    """
    Example of usage
    """
    points = batman_trajectory()
    # points = ellipse_trajectory()
    print(points.shape)
    colors = np.arange(len(points))
    plt.figure(figsize=(15, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, c=colors, cmap='hsv')
    plt.show()
    plt.plot(points[:, 0], points[:, 1])
    plt.show()


if __name__ == '__main__':
    main()
