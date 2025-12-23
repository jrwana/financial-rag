from pydantic_settings import BaseSettings
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

class Settings(BaseSettings):
    # Environment
    ENV: Literal["local", "prod"] = "local"

    # Embeddings
    EMBEDDINGS_PROVIDER: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: Literal["cuda", "cpu"] = "cuda"

    # Paths
    # INDEX_PATH: str = "./.data/index"
    # INDEX_PATH: str = f"./.data/index/{ENV}/{EMBEDDINGS_MODEL}"
    DOCS_PATH: str = "./.data"

    @property
    def index_path(self) -> str:
         # e.g., ".data/index/local/all-MiniLM-L6-v2"
         return f"./.data/index/{self.ENV}/{self.EMBEDDINGS_MODEL.replace('/', '_')}"

    # RAG settings
    DEFAULT_K: int = 4
    DEFAULT_MODEL: str = "gpt-4o-mini"

    # API Keys
    OPENAI_API_KEY: str = ""
    API_KEY: str = ""        # For /query auth in prod
    ADMIN_API_KEY: str = ""  # For /ingest auth in prod

    # Other
    RATE_LIMIT_PER_MINUTE: int = 60  # requests per mimute

    # Comma-separated origins, e.g., "https://app.example.com,https://admin.example.com"
    CORS_ORIGINS: str = ""  # Empty = use defaults based on ENV

    class Config:
          env_file = ".env"


settings = Settings()