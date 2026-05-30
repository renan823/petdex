import numpy as np

from processing.features.edges import EdgeFeatureExtractor
from processing.filters.canny import CannyFilter
from utils.colors import luminance
from domain import FeatureVector
from processing.features.hu_moments import HuMomentsFeatureExtractor
from processing.features.hog import HoGFeatureExtractor
from processing.features.haralick import HaralickFeatureExtractor
from processing.features.colors import ColorsFeatureExtractor
from processing.features.intensity import IntensityFeatureExtractor

class FeatureService:
    @staticmethod
    def extract(img: np.ndarray) -> FeatureVector:
        # Arestas para hu e edges
        edges = CannyFilter.apply(luminance(img))

        i_hist = IntensityFeatureExtractor.apply(img)
        c_stat = ColorsFeatureExtractor.apply(img)
        haralick = HaralickFeatureExtractor.apply(img)
        hog = HoGFeatureExtractor.apply(img)
        edge_i = EdgeFeatureExtractor.apply(edges)
        moments = HuMomentsFeatureExtractor.apply(edges)

        return FeatureVector(i_hist, c_stat, haralick, hog, edge_i, moments)
