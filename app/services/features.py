import numpy as np
from scipy.ndimage import label

from services.filters import CannyFilter, SobelFilter
from utils.operations import convolve, glcm
from utils.colors import luminance
from domain import Feature, FeatureVector
from services.histogram import Histogram


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
            skew = np.mean((chan - mean) ** 3) / (std**3 + 1e-6)

            features.extend([mean, std, skew])

        return np.array(features)


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

        # Desvios padrçao
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

        return np.array([max_p, corr, contr, energy, homog, entropy])



class EdgeFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        # Aplicar canny
        img = luminance(img)
        edges = CannyFilter.apply(img)

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
        


class HoGFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        img = luminance(img)
        B = 8

        # Sobel e gradientes
        sx, sy = SobelFilter.kernel()
        gx = convolve(img, sx)
        gy = convolve(img, sy)

        # Magnitude e angulo
        mag = np.hypot(gx, gy)
        ang = np.mod(np.arctan2(gy, gx), 2*np.pi)

        # Bins do hist
        bin_size = 2*np.pi / B
        
        bin_idx = np.floor(ang / bin_size).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, B-1)

        # histograma dos gradientes
        hog = np.bincount(
            bin_idx.ravel(),
            weights=mag.ravel(),
            minlength=B
        )
        
        return hog / np.linalg.norm(hog) + 1e-10 

class FeatureService:
    @staticmethod
    def extract(img: np.ndarray) -> FeatureVector:
        i_hist = IntensityHistogramFeatureExtractor.apply(img)
        c_stat = ColorStatisticsFeatureExtractor.apply(img)
        haralick = HaralickFeatureExtractor.apply(img)
        hog = HoGFeatureExtractor.apply(img)
        edges = EdgeFeatureExtractor.apply(img)

        return FeatureVector(i_hist, c_stat, haralick, hog, edges)
