import scipy.io
import numpy as np


def main():
    mat = scipy.io.loadmat('data/calibration_dataset.mat')
    pts1 = np.array(mat['mA'], dtype=np.float)[:, :, np.newaxis]
    pts2 = np.array(mat['mB'], dtype=np.float)[:, :, np.newaxis]
    pts3 = np.array(mat['mC'], dtype=np.float)[:, :, np.newaxis]

    np.set_printoptions(suppress=True, precision=10)

    dev1 = dev2 = dev3 = 0.0
    for i in range(24):
        dev1 += np.sqrt(np.var(pts1[i * 10:(i + 1) * 10] * 1000, axis=0))
        dev2 += np.sqrt(np.var(pts2[i * 10:(i + 1) * 10] * 1000, axis=0))
        dev3 += np.sqrt(np.var(pts3[i * 10:(i + 1) * 10] * 1000, axis=0))

    dev1 /= 24
    dev2 /= 24
    dev3 /= 24

    dev = (dev1 + dev2 + dev3) / 3

    print("Coordinatewise deviation: ")
    print(dev)

    print("Average deviation")
    print(np.sum(dev) / 3)


if __name__ == '__main__':
    main()
