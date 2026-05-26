import io

from flask import Blueprint, send_file, abort, request, render_template, redirect
from imageio.v3 import imread

from services.features import FeatureService
from services.storage import StorageService
from database.repository import DataRepository

main_bp = Blueprint(
    'main_pages', 
    __name__, 
    template_folder='../templates' 
)

storage = None
repository = None

def init_server():
    global storage, repository

    repository = DataRepository()
    storage = StorageService()


@main_bp.route("/", methods=["GET", "POST"])
def index():
    if not repository:
        return abort(500, description="Services not initialized.")
    
    if request.method == "GET":
        return render_template("index.html")

    # Recebe imagem
    if 'input_image' not in request.files:
        return redirect(request.url)
            
    file = request.files['input_image']
    if file.filename == '':
        return redirect(request.url)

    # Extrair caracteristicas
    try:
        content = file.read()
        img = imread(content)

        # Buscar e retornar
        features = FeatureService.extract(img)
        results = repository.search_pet_by_features(features, 10)

        return render_template("index.html", results=results)
    except Exception as e:
        return abort(500, description=str(e))


@main_bp.route("/images/<string:image_path>", methods=["GET"])
def get_image(image_path):
    if not storage:
        return abort(500, description="Services not initialized.")
        
    try:
        content = storage.read(image_path)
        image = io.BytesIO(content)

        return send_file(image, mimetype="image/jpg")
    except Exception as e:
        return abort(500, description=str(e))