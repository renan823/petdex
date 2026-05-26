import pandas as pd
import imageio.v3 as io
import dask.dataframe as dd

from domain import ImageInsertRecord
from services.features import FeatureService
from database.repository import DataRepository
from services.storage import StorageService


_GLOBAL_REPO: DataRepository | None = None
_GLOBAL_STORAGE: StorageService | None = None


def _process_chunk_pipeline(df_chunk: pd.DataFrame) -> pd.Series:
    if _GLOBAL_REPO is None or _GLOBAL_STORAGE is None:
        raise RuntimeError("Failed to initalize bulk loader")

    records: list[ImageInsertRecord] = []
    
    for _, row in df_chunk.iterrows():
        try:
            filename_str = str(row["filename"])
            
            # Ler imagem e extrair features
            img = io.imread(f'{row["basepath"]}/{filename_str}')
            features = FeatureService.extract(img)

            # Salvar no storage (uuid)
            full_path = f'{row["basepath"]}/{filename_str}'
            with open(full_path, 'rb') as f:
                content = f.read()
                
            url = _GLOBAL_STORAGE.save(content, filename_str)
            records.append(ImageInsertRecord(
                name=str(row["class_name"]),
                features=features,
                url=url
            ))
            
        except Exception as e:
            print(f"Failed to process image {row['filename']}: {e}")
    
    if records:
        _GLOBAL_REPO.insert_pets(records) 
        return pd.Series([len(records)])
        
    return pd.Series([0])


class DatasetBulkLoader:
    def __init__(self, storage: StorageService, repo: DataRepository):
        global _GLOBAL_REPO, _GLOBAL_STORAGE
        self.storage = storage
        self.repo = repo
        
        _GLOBAL_REPO = repo
        _GLOBAL_STORAGE = storage


    def load(self, basepath: str) -> int:
        # Achar base path e buscar metadata.csv
        df = pd.read_csv(f"{basepath}/metadata.csv")
        df["basepath"] = basepath
        
        # Dividir lotes de inserção
        ddf = dd.from_pandas(df, chunksize=30)
        meta = pd.Series([], dtype="int")
        
        results = ddf.map_partitions(
            _process_chunk_pipeline, 
            meta=meta,
        )

        # Mapear cada imagem de cada lote
        # Inserir um lote no repo
        inserted = results.compute(scheduler='threads')
        return int(inserted.sum())