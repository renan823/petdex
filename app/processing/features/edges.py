import numpy as np
from scipy.ndimage import label

from domain import Feature

class EdgeFeatureExtractor:
    @staticmethod
    def apply(edges: np.ndarray) -> Feature:
        # Métricas e features
        edge_density = np.count_nonzero(edges) / edges.size

        # Componentes conectados
        labels, n_components = label(edges)

        if n_components > 0:
            sizes = np.bincount(labels.ravel())[1:]
        
            mean_size = sizes.mean()
            std_size = sizes.std()
            max_size = sizes.max()
            min_size = sizes.min()
        
            # componente dominante
            largest_ratio = max_size / sizes.sum()
        else:
            mean_size = 0.0
            std_size = 0.0
            max_size = 0.0
            min_size = 0.0
            largest_ratio = 0.0
        
        return np.array([
            edge_density,
            n_components,
            mean_size,
            std_size,
            max_size,
            min_size,
            largest_ratio,
        ], dtype=np.float32)