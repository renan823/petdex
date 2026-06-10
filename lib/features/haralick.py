import numpy as np

from lib.utils.operations import glcm
from lib.utils.colors import luminance
from lib.domain import Feature


'''
Features relacionadas à textura.

Aplica o descritor de Haralick 
GLCM (Gray Level Coocurrence matrix) da imagem.

Estatísticas são isada para montar as features.
'''
class HaralickFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)
        img = img.astype(np.uint8)
       
        # Matriz de coocorrência
        M = glcm(img, (0, 1))

        # Otimização dos índices
        ridx, cidx = np.indices(M.shape)

        # Médias
        mean_r = np.sum(ridx * M)
        mean_c = np.sum(cidx * M)

        # Desvios padrão
        var_r = np.sum((ridx - mean_r) ** 2 * M)
        var_c = np.sum((cidx - mean_c) ** 2 * M)
        
        std_r = np.sqrt(var_r)
        std_c = np.sqrt(var_c)
        
        # Descritores (otimizados pq tava lento)
        max_p = M.max()
        corr = np.sum(((ridx - mean_r) * (cidx - mean_c) * M)) / (std_r * std_c)
        contr = np.sum((ridx - cidx) ** 2 * M)
        energy = np.sum(M * M)
        homog = np.sum(M / (1 + np.abs(ridx - cidx)))
        mask = M > 0
        entropy = -np.sum(M[mask] * np.log2(M[mask]))

        har = np.array([max_p, corr, contr, energy, homog, entropy])
        return har / (np.linalg.norm(har) + 1e-10)

