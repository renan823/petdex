import numpy as np
from numpy.typing import NDArray

Feature = NDArray[np.float32]

class ImageResult:
    def __init__(self, name: str, url: str, distance: float, features: Feature):
        self.name = name
        self.url = url
        self.distance = distance
        self.features = features


class ImageRecord:
    def __init__(self, name: str, url: str, features: Feature):
        self.name = name
        self.url = url
        self.features = features