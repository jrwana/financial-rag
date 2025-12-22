"""
Convert text chunks into vectors and store them in FAISS
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from src.config import settings
import json

# default paths
INDEX_DIR = settings.index_path
DEFAULT_MODEL = settings.EMBEDDINGS_MODEL


def get_embeddings(
        provider: str | None = None,
        model_name: str | None = None
        ) -> Embeddings:
    """
    Initialize embedding based on provider

    Lazy imports torch/sentence-transformers so prod deployments
    using OpenAI don't need them installed
    """
    provider = provider or settings.EMBEDDINGS_PROVIDER
    model_name = model_name or settings.EMBEDDINGS_MODEL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
        )

    elif provider == "sentence_transformers":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": settings.DEVICE},
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_index(chunks: list, model_name: str = DEFAULT_MODEL) -> FAISS:
    """Create FAISS index from document chunks"""
    embeddings = get_embeddings(model_name=model_name)

    print(f"Creating index from {len(chunks)} chunks...")
    index = FAISS.from_documents(chunks, embeddings)
    print("Index created")

    return index


def save_index(index: FAISS, index_dir: str = INDEX_DIR):
    """Save FAISS index to disk"""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index.save_local(index_dir)
    print(f"Index saved to {index_dir}")

    metadata = {
        "provider": settings.EMBEDDINGS_PROVIDER,
        "model": settings.EMBEDDINGS_MODEL
    }
    Path(index_dir, "metadata.json").write_text(json.dumps(metadata))


def load_index(index_dir: str = INDEX_DIR, model_name: str = DEFAULT_MODEL) -> FAISS:
    """Load FAISS index from disk"""
    if not Path(index_dir).exists():
        raise FileNotFoundError(f"Index not found at {index_dir}. Run create_index first.")

    # check if current settings matches metadata
    metadata_path = Path(index_dir) / "metadata.json"
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["provider"] != settings.EMBEDDINGS_PROVIDER:
            raise ValueError(f"Index was created with {saved['provider']}, but current provider is {settings.EMBEDDINGS_PROVIDER}")
        if saved["model"] != settings.EMBEDDINGS_MODEL:
            raise ValueError(f"Index was created with {saved['model']}, but current model is {settings.EMBEDDINGS_MODEL}")

    embeddings = get_embeddings(model_name=model_name)
    index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    print(f"Index loaded from {index_dir}")
    return index


if __name__ == "__main__":
    from src.ingestion import ingest

    # build and save index
    chunks = ingest()
    index = create_index(chunks)
    save_index(index)

    # quick test
    results = index.similarity_search("revenue", k=2)
    print(f"\nTest query results: {len(results)} docs found")





