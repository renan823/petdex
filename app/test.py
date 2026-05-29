import imageio.v3 as io

from services.features import FeatureService

def main():
    img = io.imread("./data/billy_franzen/00219.jpg")
    print(len(FeatureService.extract(img).data))


if __name__ == "__main__":
    main()