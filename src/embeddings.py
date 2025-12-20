"""
Convert text chunks into vectors and store them in FAISS
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# default paths
INDEX_DIR = "./index"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_MODEL):
    """Initialize embedding model"""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_index(chunks: list, model_name: str = DEFAULT_MODEL) -> FAISS:
    """Create FAISS index from document chunks"""
    embeddings = get_embeddings(model_name)

    print(f"Creating index from {len(chunks)} chunks...")
    index = FAISS.from_documents(chunks, embeddings)
    print("Index created")

    return index


def save_index(index: FAISS, index_dir: str = INDEX_DIR):
    """Save FAISS index to disk"""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    index.save_local(index_dir)
    print(f"Index saved to {index_dir}")


def load_index(index_dir: str = INDEX_DIR, model_name: str = DEFAULT_MODEL) -> FAISS:
    """Load FAISS index from disk"""
    if not Path(index_dir).exists():
        raise FileNotFoundError(f"Index not found at {index_dir}. Run create_index first.")
    embeddings = get_embeddings(model_name)
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





