import numpy as np

from domain import Feature


class ColorsFeatureExtractor:
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