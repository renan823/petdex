import numpy as np


'''
Conversão de RGB para intensidade.
'''
def luminance(img: np.ndarray) -> np.ndarray:
    lum = [0.2126, 0.7152, 0.0722]
    return np.dot(img[..., :3], lum)