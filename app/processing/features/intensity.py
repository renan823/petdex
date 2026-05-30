import numpy as np

from processing.histogram import Histogram
from domain import Feature


class IntensityFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hist = Histogram(img, 8, bw=True)
        return hist.data



