import numpy as np


class LaplacianFilter:
    @staticmethod
    def kernel():
        return np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ])