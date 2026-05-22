import numpy as np

from app.services.histogram import Histogram

Feature = np.ndarray

class HistogramFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hist = Histogram(img, 8)
        return hist.data


class CannyFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
       
        
        return img

