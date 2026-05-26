import uuid
import shutil
from pathlib import Path

class StorageService:
    def __init__(self):
        self.base_dir = Path("storage").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        

    def save(self, content: bytes, filename: str) -> str:
        extension = Path(filename).suffix.lower()
        if not extension:
            extension = ".jpg"
        
        # Nome único com UUID
        unique_name = f"{uuid.uuid4()}{extension}"
        file_path = self.__get_safe_path(unique_name)
        
        # Grava bytes
        with open(file_path, "wb") as f:
            f.write(content)
        
        return unique_name


    def read(self, filename: str) -> bytes:
        file_path = self.__get_safe_path(filename)
            
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File '{filename}' not found")
    
        with open(file_path, "rb") as f:
            return f.read()

    
    def clear(self):
        if self.base_dir == Path("/").resolve() or self.base_dir == Path.home().resolve():
            raise Exception("Failed to clear storage")
        
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
                
        self.base_dir.mkdir(parents=True, exist_ok=True)


    def __get_safe_path(self, filename: str) -> Path:
        file_path = (self.base_dir / filename).resolve()

        # Evita path traversal
        if not file_path.is_relative_to(self.base_dir):
            raise Exception("Invalid path - Not allowed")
                    
        return file_path
    