import numpy as np
import sklearn.cluster


class BoVW:
    def __init__(self, extractor, size: int, n_clusters: int):
        self.size = size
        self.n_clusters = n_clusters
        self.extractor = extractor
        self.regions: list[np.ndarray] = []
        self.kmeans = sklearn.cluster.KMeans(
            n_clusters=n_clusters,
            n_init=3,
            init='random'
        )
        
    '''
    Divisão da imagem em regiões para aplicação da
    extração de features.
    '''
    def __get_region(self, img: np.ndarray) -> np.ndarray:
        h, w, _ = img.shape
        r = self.size // 2
        
        regions = []
        for i in np.arange(r, h-r, self.size):
            for j in np.arange(r, w-r, self.size):
                region = img[i-r:i+r, j-r:j+r]
                regions.append(region)
                
        return np.array(regions)
    
    
    '''
    Adiciona as regiões de uma imagem na lista
    de regiões, posteriormente usada nas features.
    '''
    def add_image(self, img: np.ndarray):
        # Gerar regiões
        self.regions.append(self.__get_region(img))
    
    
    '''
    Criação do dicionário de visula words.
    Com a lista de regiões armazenada, gera
    as features para cada pedaço das imagens.
    Usa kmeans para aplicar cluster nos resultados.
    '''
    def compute(self):
        if len(self.regions) == 0:
            return
        
        # Extração das features e aplicação do kmeans
        regions = np.concatenate(self.regions)
        
        features = np.array([self.extractor(r) for r in regions])
        self.kmeans.fit(features)
        

    '''
    Extração do histograma de caracteristicas
    de uma nova imagem após sua clusterização
    no dicionário de visual words.
    '''
    def apply(self, img: np.ndarray) -> np.ndarray:
        # Validação se foi treinado
        if not hasattr(self.kmeans, "cluster_centers_"):
            raise RuntimeError("BoVW not trained")
        
        # Extração da região e features
        region = self.__get_region(img)
        feature = [self.extractor(r) for r in region]
        
        # Criação do histograma de visual words
        clusters = self.kmeans.predict(feature)
        H, _ = np.histogram(clusters, np.arange(0, self.n_clusters + 1))
        
        return H.astype(float) / H.sum()