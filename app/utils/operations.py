import numpy as np

'''
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

def non_max_suppression(mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    suppressed = np.zeros_like(mag)
    h, w = mag.shape
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = theta[i, j]
            q = 255
            r = 255
    
            # 0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
    
            # 45
            elif 22.5 <= angle < 67.5:
                q = mag[i - 1, j + 1]
                r = mag[i + 1, j - 1]
    
            # 90
            elif 67.5 <= angle < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
    
            # 135
            elif 112.5 <= angle < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
    
            if mag[i, j] >= q and mag[i, j] >= r:
                suppressed[i, j] = mag[i, j]

    return suppressed

def double_threshold(img: np.ndarray, low=0.05, high=0.15) -> np.ndarray:
    strong = 255
    weak = 75
    
    result = np.zeros_like(img)
    
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where(
        (img >= low) &
        (img < high)
    )
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result

    
def hysteresis(img: np.ndarray, weak=75, strong=255) -> np.ndarray:
    h, w = img.shape
    result = np.zeros_like(img)
        
    # Bordas fortes (iniciais)
    strong_i, strong_j = np.where(img == strong)
    queue = list(zip(strong_i, strong_j))
        
    # Coloca as fortes no resultado
    result[strong_i, strong_j] = strong
        
    # Busca vizinhança
    while len(queue) > 0:
        curr_i, curr_j = queue.pop(0)
            
        for di in [-1, 0, 1]:
             for dj in [-1, 0, 1]:
                ni, nj = curr_i + di, curr_j + dj
                    
                if 0 <= ni < h and 0 <= nj < w:
                    # Vizinho fraco vira forte
                    if img[ni, nj] == weak and result[ni, nj] != strong:
                        result[ni, nj] = strong
                        queue.append((ni, nj))
    
    return result