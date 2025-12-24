"""
 Load financial PDFs and split them into chunks for RAG pipeline
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(data_dir: str = "./data") -> list:
    """Load all PDFs from the data directory"""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_files = list(data_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())
        print(f"Loaded: {pdf_file.name}")

    return documents


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def ingest(data_dir: str = "./data") -> list:
    """Main ingestion pipeline"""
    documents = load_pdfs(data_dir)
    chunks = split_documents(documents)

    # Add chunk_id to metadata
    for i, chunk in enumerate(chunks):
        if "chunk_id" not in chunk.metadata:
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["chunk_id"] = f"{source}_{i}"

    return chunks


if __name__ == "__main__":
    chunks = ingest()
    # preview first chunk
    if chunks:
        print(f"\nSample chunk:\n{chunks[0].page_content[:300]}...")
