import numpy as np

def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    h, w = img.shape
        
    result = np.zeros_like(img)
        
    kernel = np.flip(kernel)
    kh, kw = kernel.shape
        
    a = (kh - 1) // 2
    b = (kw - 1) // 2 
        
    # Ignorar a borda
    for i in range(a, h - a):
        for j in range (b, w - b):
            # Vizinhança do ponto (com mult. ponto a ponto)
            neigh = img[i-a:i+a+1, j-b:j+b+1]
            result[i, j] = (kernel * neigh).sum()
        
    return result


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
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
    
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
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == weak:
                neighborhood = img[i-1:i+2, j-1:j+2]
    
                if np.any(neighborhood == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img