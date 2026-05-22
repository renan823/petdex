import imageio.v3 as io
import matplotlib.pyplot as plt

from services.filters import CannyFilter
from utils.colors import luminance

def main():
    img = io.imread("data/bruce/00195.jpg")
    
    edges = CannyFilter.apply(luminance(img))

    plt.subplot(1, 2, 1)
    plt.imshow(edges)

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    
    plt.show()

if __name__ == "__main__":
    main()
