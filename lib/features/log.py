import numpy as np

from lib.filters.gaussian import GaussianFilter
from lib.filters.laplacian import LaplacianFilter
from lib.utils.operations import convolve
from lib.utils.colors import luminance
from lib.domain import Feature


'''
Features relacionadas à

Aplica a LoG (Laplacian of Gaussian) na imagem
e extrai algumas estatísticas para construir
o vetor de features.
'''
class LoGFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)

        G = GaussianFilter.kernel(5, 2)
        L = LaplacianFilter.kernel()

        log = convolve(convolve(img, G), L)

        # Features básicas
        log_f = np.array([
            log.mean(),
            log.std(),
            log.var(),
            log.max(),
        ])
        
        return log_f / (np.linalg.norm(log_f) + 1e-10)