import numpy as np

from app.utils.colors import luminance


'''
Classe para o histograma.
Pode ser usado para imagens coloridos ou preto/branco.
'''
class Histogram:
    def __init__(self, img: np.ndarray, size: int, bw=True) -> None:
        if bw:
            self.data = self.__bw_hist(img, size)
        else:
            self.data = self.__color_hist(img, size)
        

    def __hist(self, img: np.ndarray, size: int) -> np.ndarray:
        h, _ = np.histogram(img, bins=range(0, 256, 256//size), density=True)
        return h


    def __bw_hist(self, img: np.ndarray, size: int) -> np.ndarray:
        return self.__hist(luminance(img), size)


    def __color_hist(self, img: np.ndarray, size: int) -> np.ndarray:
        r = self.__hist(img[..., 0], size)
        g = self.__hist(img[..., 1], size)
        b = self.__hist(img[..., 2], size)
           
        return np.concatenate([r, g, b]) / 3
