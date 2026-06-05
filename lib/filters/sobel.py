import numpy as np

class SobelFilter:
    @staticmethod
    def kernel() -> tuple[np.ndarray, np.ndarray]:
        x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        return x, y
