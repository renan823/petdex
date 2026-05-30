import numpy as np

Feature = np.ndarray

class FeatureVector:
    def __init__(
        self, 
        i_hist: Feature,
        c_stat: Feature,
        haralick: Feature,
        hog: Feature,
        #edges: Feature
    ):
        self.data: list[float] = []
        
        self.data.extend(i_hist.tolist())
        self.data.extend(c_stat.tolist())
        self.data.extend(haralick.tolist())
        self.data.extend(hog.tolist())
        #self.data.extend(edges.tolist())

class ImageRecord:
    def __init__(self, id: int, name: str, url: str, distance: float):
        self.id = id
        self.name = name
        self.url = url
        self.distance = distance


class ImageInsertRecord:
    def __init__(self, name: str, url: str, features: FeatureVector):
        self.name = name
        self.url = url
        self.features = features