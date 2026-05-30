from pymilvus import MilvusClient, DataType

from settings import DB_NAME


'''
Criação da base de dados vetorial, usando o Milvus.
A coleção armazena metadados da foto e as features.

Para indexar as features, usei o HNSW (Hierarquical Navigable Small Worlds).
Documentação: https://milvus.io/docs/hnsw.md
'''
class DatabaseMigrator:
    def apply(self):
        client = MilvusClient(DB_NAME)

        # Aplicar migrações
        self.__migrate_pet_collection(client)
    
        client.close()

    def __migrate_pet_collection(self, client: MilvusClient):
        # Remove se já existe e recria
        if client.has_collection("pets"):
            client.drop_collection("pets")

        # Esquema das "tabelas"
        schema = client.create_schema()
        
        # As features são armazenadas em um vetor só, sempre na mesma ordem
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="name", datatype=DataType.VARCHAR)
        schema.add_field(field_name="url", datatype=DataType.VARCHAR)
        schema.add_field(field_name="features", datatype=DataType.FLOAT_VECTOR, dim=30)

        # Índice HNSW
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="features", 
            index_type="HNSW",
            index_name="features_index",
            metric_type="L2", # Euclidiana
            params={ "M": 64, "efConstruction": 100 } 
        )

        # Coleção "pets"
        client.create_collection(
            collection_name="pets",
            schema=schema,
            index_params=index_params,
            metric_type="L2",
        )

