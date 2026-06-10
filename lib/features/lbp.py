import numpy as np
from skimage.feature import local_binary_pattern

from lib.utils.colors import luminance
from lib.domain import Feature


'''
Features relacionadas à textura.

Aplica o LBP na imagem e retorna um
histograma dos padrões encontrados.
'''
class LBPFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)
        img = img.astype(np.uint8)
        
        radius = 1
        points = 8 * radius
        lbp = local_binary_pattern(img, R=radius, P=points, method="uniform")
        
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, points + 3),
            range=(0, points + 2)
        )
        
        hist = hist.astype(np.float32)
        hist /= hist.sum()
        
        return hist


        