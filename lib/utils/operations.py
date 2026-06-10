from collections import deque

import numpy as np

'''
Função de convolução.

Melhoria com sliding window:
https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
'''
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    kernel = np.flip(kernel)
        
    kh, kw = kernel.shape
    h, w = img.shape
        
    a = (kh - 1) // 2
    b = (kw - 1) // 2
        
    # Cria a sliding window nos pixels
    windows = np.lib.stride_tricks.sliding_window_view(img, (kh, kw))
        
    # Aplica a convolução considerando os vizinhos da janela
    conv_result = np.sum(windows * kernel, axis=(-2, -1))
    result = np.zeros_like(img)
    result[a:h-a, b:w-b] = conv_result
        
    return result

'''
Matriz de coocorrência para haralick.
Feita com auxílio de IA (estava MUITO lento)
'''
def glcm(img: np.ndarray, d: tuple[int, int]):
    h, w = img.shape
    dy, dx = d

    # Região válida da imagem original
    y0 = max(0, -dy)
    y1 = min(h, h - dy)

    x0 = max(0, -dx)
    x1 = min(w, w - dx)

    p1 = img[y0:y1, x0:x1]
    p2 = img[y0 + dy:y1 + dy, x0 + dx:x1 + dx]

    M = np.zeros((256, 256), dtype=np.uint32)

    np.add.at(M, (p1.ravel(), p2.ravel()), 1)

    return M / M.sum()
