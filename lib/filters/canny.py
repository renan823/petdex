import numpy as np

import lib.utils.operations as Op
from lib.filters.gaussian import GaussianFilter
from lib.filters.sobel import SobelFilter

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