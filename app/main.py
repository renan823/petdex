import streamlit as st
from imageio.v3 import imread
from PIL import Image

from services.loader import DatasetBulkLoader
from database.migrate import DatabaseMigrator
from services.features import FeatureService
from database.repository import DataRepository
from services.storage import StorageService

def render_search(
    repository: DataRepository,
    storage: StorageService
):
    uploaded_file = st.file_uploader(
        "Selecione uma imagem",
        type=["jpg", "jpeg", "png"],
        key="search_image"
    )

    if uploaded_file is None:
        st.info("Selecione uma imagem para iniciar a busca.")
        return

    query_image = Image.open(uploaded_file)

    uploaded_file.seek(0)
    img = imread(uploaded_file.read())

    with st.spinner("Buscando imagens semelhantes..."):
        features = FeatureService.extract(img)
        results = repository.search_pet_by_features(features, 10)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Consulta")
        st.image(query_image, use_container_width=True)

    with right:
        st.subheader("Resultados")

        cols = st.columns(3)

        for i, result in enumerate(results):
            content = storage.read(result.url)

            cols[i % 3].image(
                content,
                caption=result.url,
                use_container_width=True
            )

def render_migration():
    st.warning("Esta operação remove todos os dados atuais.")

    confirm = st.checkbox(
        "Entendo que os dados serão apagados",
        key="migration_confirm"
    )

    if confirm and st.button(
        "Executar Migração",
        type="primary"
    ):
        with st.spinner("Migrando banco..."):
            storage = StorageService()
            migrator = DatabaseMigrator()

            storage.clear()
            migrator.apply()

        st.success("Migração concluída.")


def render_load():
    dataset_path = st.text_input("Diretório do dataset")

    if st.button("Carregar Dataset"):
        if not dataset_path:
            st.error("Informe o diretório.")
            return

        with st.spinner("Importando dataset..."):
            storage = StorageService()
            repository = DataRepository()

            loader = DatasetBulkLoader(
                storage,
                repository
            )

            items = loader.load(dataset_path)

        st.success(
            f"{items} imagens carregadas."
        )


def run():
    repository = DataRepository()
    storage = StorageService()

    st.set_page_config(
        page_title="Petdex",
        layout="wide"
    )

    st.title("Petdex")

    tab_search, tab_load, tab_migrate = st.tabs(
        [
            "Busca",
            "Carregar dados",
            "Migrar base de dados"
        ]
    )

    with tab_search:
        render_search(
            repository,
            storage
        )

    with tab_load:
        render_load()

    with tab_migrate:
        render_migration()

if __name__ == "__main__":
    run()