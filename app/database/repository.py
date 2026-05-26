from pymilvus.milvus_client import MilvusClient

from domain import FeatureVector, ImageInsertRecord, ImageRecord
from settings import DB_NAME


class DataRepository:
    def __init__(self):
        self.client = MilvusClient(DB_NAME)
        if not self.client.has_collection("pets"):
            raise Exception("Collection not found. Did you run migration commands?")

        self.client.load_collection("pets")

    def close(self):
        self.client.close()


    def insert_pets(self, images: list[ImageInsertRecord]) -> int:
        data = [{"name": i.name, "features": i.features.data, "url": i.url} for i in images]
        
        res = self.client.insert(
            collection_name="pets",
            data=data
        )

        return res["insert_count"]
        

    def search_pet_by_features(self, vec: FeatureVector, limit: int) -> list[ImageRecord]:
        res = self.client.search(
            collection_name="pets",
            anns_field="features",
            data=[vec.data],
            limit=limit,
            output_fields=["name", "url"],
            search_params={"metric_type": "L2"}
        )

        matches = []
        if len(res) == 0:
            return matches

        # Parse dos resultados
        hits = res[0]
        for hit in hits:
            matches.append(ImageRecord(
                id=hit.get("id") or hit["entity"].get("id"),
                name=hit["entity"]["name"],
                url=hit["entity"]["url"],
                distance=hit["distance"]
            ))

        return matches