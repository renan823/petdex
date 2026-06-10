import numpy as np

from lib.domain import Feature, ImageRecord, ImageResult


'''
Base de dados simplificada que
armazena os registros de imagens e
realiza a busca por similaridade nos
vetores de features.

Usa distância euclidiana L2.
'''
class Database:
    def __init__(self):
        self.data: list[ImageRecord] = []
    
    
    def insert(self, rec: ImageRecord):
        if not rec:
            return
        
        self.data.append(rec)
        
        
    def search(self, vec: Feature, limit=10) -> list[ImageResult]:
        results = []

        for rec in self.data:
            dist = self.distance(vec, rec.features)

            results.append(
                ImageResult(
                    distance=dist,
                    name=rec.name,
                    url=rec.url,
                    features=rec.features
                )
            )

        results.sort(key=lambda r: r.distance)
        return results[:limit]
    
    
    def distance(self, a: Feature, b: Feature) -> float:
        return np.linalg.norm(a - b)