import numpy as np

from domain import Feature

'''
Momentos de Hu.

https://pt.wikipedia.org/wiki/Momentos_invariantes_de_uma_imagem
'''
class HuMomentsFeatureExtractor:
    @staticmethod
    def apply(img: np.ndarray) -> Feature:
        # Otimização e evita calcular dnv
        xs, ys, values = active_pixels(img)
        
        # Momentos base
        m00 = moment(xs, ys, values, 0, 0)
        m10 = moment(xs, ys, values, 1, 0)
        m01 = moment(xs, ys, values, 0, 1)

        # Centro de massa
        cx = m10 / m00
        cy = m01 / m00

        # Momentos centrais
        u00 = central_moment(xs, ys, values, 0, 0, cx, cy)
        u11 = central_moment(xs, ys, values, 1, 1, cx, cy)
        u20 = central_moment(xs, ys, values, 2, 0, cx, cy)
        u02 = central_moment(xs, ys, values, 0, 2, cx, cy)
        u21 = central_moment(xs, ys, values, 2, 1, cx, cy)
        u12 = central_moment(xs, ys, values, 1, 2, cx, cy)
        u30 = central_moment(xs, ys, values, 3, 0, cx, cy)
        u03 = central_moment(xs, ys, values, 0, 3, cx, cy)

        # Momentos normalizados
        n11 = normalized_moment(u00, u11, 1, 1)
        n20 = normalized_moment(u00, u20, 2, 0)
        n02 = normalized_moment(u00, u02, 0, 2)
        n21 = normalized_moment(u00, u21, 2, 1)
        n12 = normalized_moment(u00, u12, 1, 2)
        n30 = normalized_moment(u00, u30, 3, 0)
        n03 = normalized_moment(u00, u03, 0, 3)

        # Momentos invariantes
        I1 = n20 + n02
        I2 = ((n20 - n02) ** 2) + (4 * (n11 ** 2))
        I3 = ((n30 - 3 * n12) ** 2) + ((3 * n21 - n03) ** 2)
        I4 = ((n30 + n12) ** 2) + ((n21 + n03) ** 2)
        I5 = (
            (n30 - 3 * n12)
            * (n30 + n12)
            * (((n30 + n12) ** 2) - (3 * ((n21 + n03) ** 2)))
        ) + (
            (3 * n21 - n03)
            * (n21 + n03)
            * ((3 * ((n30 + n12) ** 2)) - ((n21 + n03) ** 2))
        )
        I6 = (
            (n20 - n02)
            * (((n30 + n12) ** 2) - ((n21 + n03) ** 2))
        ) + (
            4 * n11 * (n30 + n12) * (n21 + n03)
        )
        I7 = (
            (3 * n21 - n03)
            * (n30 + n12)
            * (((n30 + n12) ** 2) - (3 * ((n21 + n03) ** 2)))
        ) - (
            (n30 - 3 * n12)
            * (n21 + n03)
            * ((3 * ((n30 + n12) ** 2)) - ((n21 + n03) ** 2))
        )

        moments = np.array([I1, I2, I3, I4, I5, I6, I7], dtype=float)
        for i in range(len(moments)):
            MI = moments[i]
            moments[i] = np.sign(MI) * np.log10(np.abs(MI))

        return moments


def active_pixels(img):
    ys, xs = np.nonzero(img)
    values = img[ys, xs]

    return xs.astype(np.float64), ys.astype(np.float64), values.astype(np.float64)

'''
'''
def moment(xs, ys, values, p, q):
    return np.sum((xs ** p) * (ys ** q) * values)

'''
'''
def central_moment(xs, ys, values, p, q, cx, cy):
    dx = xs - cx
    dy = ys - cy
    
    return np.sum((dx ** p) * (dy ** q) * values)

'''
'''
def normalized_moment(u00, u, p, q):
    ex = ((p + q) / 2) + 1
    return u / (u00 ** ex)
