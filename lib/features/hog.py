import numpy as np

from lib.domain import Feature
from lib.filters.sobel import SobelFilter
from lib.utils.colors import luminance
from lib.utils.operations import convolve

class HoGFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)
        B = 8

        # Sobel e gradientes
        sx, sy = SobelFilter.kernel()
        gx = convolve(img, sx)
        gy = convolve(img, sy)

        # Magnitude e angulo
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), 2*np.pi)

        # Bins do hist
        bin_size = 2*np.pi / B
        
        bin_idx = np.floor(ang / bin_size).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, B-1)

        # histograma dos gradientes
        hog = np.bincount(
            bin_idx.ravel(),
            weights=mag.ravel(),
            minlength=B
        )
        
        return hog / np.linalg.norm(hog) + 1e-10 