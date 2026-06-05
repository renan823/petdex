import numpy as np

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
