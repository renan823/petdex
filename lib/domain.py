import numpy as np
from numpy.typing import NDArray

'''
Alias para representar uma feature
'''
Feature = NDArray[np.float32]


'''
Classe que representa um resultado
da busca por similaridade.
'''
class ImageResult:
    def __init__(self, name: str, url: str, distance: float, features: Feature):
        self.name = name
        self.url = url
        self.distance = distance
        self.features = features


'''
Registro inserido na base de dados.
'''
class ImageRecord:
    def __init__(self, name: str, url: str, features: Feature):
        self.name = name
        self.url = url
        self.features = features