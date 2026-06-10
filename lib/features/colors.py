import numpy as np
from skimage.color import rgb2hsv

from lib.domain import Feature


'''
Features relacionadas à cor.

Extrai histogramas para cada canal de cor,
e gera um vetor de features concatenando esses
histogramas normalizados.

Para melhor funcionamento, as cores são
convertidas para HSV.
'''
class ColorsFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hsv = rgb2hsv(img)
       
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        
        hist_h, _ = np.histogram(
            h.ravel(),
            bins=8,
            range=(0, 1)
        )

        hist_s, _ = np.histogram(
            s.ravel(),
            bins=8,
            range=(0, 1)
        )

        hist_v, _ = np.histogram(
            v.ravel(),
            bins=8,
            range=(0, 1)
        )
        
        hist_h = hist_h.astype(np.float32)
        hist_s = hist_s.astype(np.float32)
        hist_v = hist_v.astype(np.float32)

        hist_h /= hist_h.sum()
        hist_s /= hist_s.sum()
        hist_v /= hist_v.sum()
        
        return np.concatenate([hist_h, hist_s, hist_v])