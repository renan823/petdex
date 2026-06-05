import numpy as np

from lib.histogram import Histogram
from lib.domain import Feature


class ColorsFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hist = Histogram(img, 8, bw=False)
        return hist.data