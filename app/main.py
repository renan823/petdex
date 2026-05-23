import imageio.v3 as io
import matplotlib.pyplot as plt

from services.filters import CannyFilter
from utils.colors import luminance

def main():
    img = io.imread("data/dakota/00165.jpg")
    
    edges = CannyFilter.apply(luminance(img))

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    
    plt.show()

if __name__ == "__main__":
    main()
