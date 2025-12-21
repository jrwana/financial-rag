from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Environment
    ENV: Literal["local", "prod"] = "local"

    # Embeddings
    EMBEDDINGS_PROVIDER: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: Literal["cuda", "cpu"] = "cuda"

    # Paths
    INDEX_PATH: str = "./.data/index"
    DOCS_PATH: str = "./data"

    # RAG settings
    DEFAULT_K: int = 4
    DEFAULT_MODEL: str = "gpt-4o-mini"

    # API Keys
    OPENAI_API_KEY: str = ""
    API_KEY: str = ""        # For /query auth in prod
    ADMIN_API_KEY: str = ""  # For /ingest auth in prod

    class Config:
          env_file = ".env"

settings = Settings()