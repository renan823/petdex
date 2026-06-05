import numpy as np

def luminance(img: np.ndarray) -> np.ndarray:
    lum = [0.2126, 0.7152, 0.0722]
    return np.dot(img[..., :3], lum)