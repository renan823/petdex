import argparse

from flask import Flask

from server.handler import main_bp, init_server
from services.loader import DatasetBulkLoader
from database.migrate import DatabaseMigrator
from database.repository import DataRepository
from services.storage import StorageService

app = Flask(__name__)

def migrate(args: argparse.Namespace):
    print("[Petdex] Migrating database...")
    
    storage = StorageService()
    migrator = DatabaseMigrator()

    # Limpa o storage e o db
    storage.clear()
    migrator.apply()

    print("[Petdex] Applyed.")
    

def load(args: argparse.Namespace):
    print("[Petdex] Loading data...")
    
    storage = StorageService()
    repo = DataRepository()

    loader = DatasetBulkLoader(storage, repo)
    items_loaded = loader.load(args.basepath)

    print(f"[Petdex] Applyed ({items_loaded} items).")


def classify(args: argparse.Namespace):
    pass


def server(args: argparse.Namespace):
    print("Iniciando o servidor Flask...")
    init_server()
            
    app.register_blueprint(main_bp)
    app.run(host='0.0.0.0', port=5000)


def main():
    parser = argparse.ArgumentParser(description="Petdex")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Operation modes")

    # Função de migração
    parser_migrate = subparsers.add_parser("migrate", help="Migrate database")
    parser_migrate.set_defaults(func=migrate)

    # Função de load e extração das features
    parser_load = subparsers.add_parser("load", help="Load images and extract features")
    parser_load.add_argument("basepath", type=str, help="Path to metadata.csv")
    parser_load.set_defaults(func=load)

    # Função de servidor
    parser_server = subparsers.add_parser("server", help="Start server")
    parser_server.set_defaults(func=server)

    args = parser.parse_args()
    args.func(args)
        
if __name__ == "__main__":
    main()
