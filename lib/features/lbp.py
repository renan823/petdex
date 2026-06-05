import numpy as np
from skimage.feature import local_binary_pattern

from lib.utils.colors import luminance
from lib.histogram import Histogram
from lib.domain import Feature

class LBPFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)
        img = img.astype(np.uint8)
        bin = local_binary_pattern(img, 20, 3)

        hist = Histogram(bin, 16, bw=True)
        return hist.data