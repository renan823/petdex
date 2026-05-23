import numpy as np

import utils.operations as Op

'''
Filtro Gaussiano.

Gerado usando uma matriz NxN e um valor
sigma informado.
'''
class GaussianFilter:
    @staticmethod
    def kernel(size: int, sigma: float) -> np.ndarray:
        kernel = np.zeros((size, size))

        k = size // 2
        denom = 2 * (sigma ** 2)

        for i in range(size):
            for j in range(size):
                iv = (i - k) ** 2
                jv = (j - k) ** 2
                
                kernel[i, j] = np.exp(- (iv + jv) / denom)

        return kernel / kernel.sum()


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

class CannyFilter:
    @staticmethod
    def apply(img: np.ndarray) -> np.ndarray:
        # Aplicar gaussiano
        gk = GaussianFilter.kernel(5, 0.5)
        img = Op.convolve(img, gk)

        # Sobel e gradientes
        x, y = SobelFilter.kernel()
        Gx = Op.convolve(img, x)
        Gy = Op.convolve(img, y)

        # Direção dos gradientes
        # Quão forte a aresta é, e pra onde aponta
        magnitude = np.hypot(Gx, Gy)
        magnitude = (magnitude / magnitude.max())
        theta = np.rad2deg(np.arctan2(Gy, Gx))
        theta[theta < 0] += 180

        # Suprimir gradientes e limitar arestas "inuteis"
        suppressed = Op.non_max_suppression(magnitude, theta)
        thresholded = Op.double_threshold(suppressed)
        
        # Manter arestas importantes (hysteresis)
        # "Inclui" arestas fracas que estão junto das fortes
        return Op.hysteresis(thresholded)
