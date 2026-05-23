import numpy as np

from services.histogram import Histogram

Feature = np.ndarray

class IntensityHistogramFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hist = Histogram(img, 8, bw=True)
        return hist.data


class ColorHistogramFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        hist = Histogram(img, 8, bw=False)
        return hist.data


class ColorStatisticsFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        features = []

        for i in range(3):
            chan = img[:, :, i]

            mean = np.mean(chan)
            std = np.std(chan)
            skew = np.mean((chan - mean) ** 3) / (std ** 3 + 1e-6)

            features.extend([mean, std, skew])

        return np.array(features)

    
'''
class CannyFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
       
        
        return img
'''
