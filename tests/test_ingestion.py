"""Unit tests for ingestion module"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.ingestion import load_pdfs, split_documents, ingest

# Path to fixture documents
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "docs"


class TestLoadPdfs:
    """Tests for load_pdfs function"""

    def test_raises_error_when_directory_not_found(self, tmp_path):
        """Should raise FileNotFoundError for non-existent directory"""
        fake_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_pdfs(str(fake_path))

        assert "Data directory not found" in str(exc_info.value)

    def test_returns_empty_list_when_no_pdfs(self, tmp_path):
        """Should return empty list when directory has no PDFs"""
        result = load_pdfs(str(tmp_path))

        assert result == []

    def test_handles_path_is_file_instead_of_directory(self, tmp_path):
        """Should return empty list (or handle gracefully) when path is a file"""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.touch()
        
        # Currently load_pdfs uses glob on the path. 
        # Path("file.txt").glob("*.pdf") returns empty list.
        result = load_pdfs(str(file_path))
        assert result == []

    @patch("src.ingestion.PyPDFLoader")
    def test_loads_single_pdf(self, mock_loader_class, tmp_path):
        """Should load a single PDF file"""
        # Create a fake PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        # Mock the loader
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        result = load_pdfs(str(tmp_path))

        assert len(result) == 1
        mock_loader_class.assert_called_once_with(str(pdf_file))

    @patch("src.ingestion.PyPDFLoader")
    def test_loads_multiple_pdfs(self, mock_loader_class, tmp_path):
        """Should load multiple PDF files"""
        # Create fake PDF files
        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.pdf").touch()
        (tmp_path / "doc3.pdf").touch()

        # Mock the loader
        mock_doc = Mock()
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        result = load_pdfs(str(tmp_path))

        assert len(result) == 3
        assert mock_loader_class.call_count == 3

    @patch("src.ingestion.PyPDFLoader")
    def test_ignores_non_pdf_files(self, mock_loader_class, tmp_path):
        """Should only load PDF files, ignoring other file types"""
        # Create mixed files
        (tmp_path / "document.pdf").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "image.png").touch()

        mock_doc = Mock()
        mock_loader = Mock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        result = load_pdfs(str(tmp_path))

        assert len(result) == 1
        mock_loader_class.assert_called_once()


class TestSplitDocuments:
    """Tests for split_documents function"""

    @patch("src.ingestion.RecursiveCharacterTextSplitter")
    def test_splits_documents_with_default_params(self, mock_splitter_class):
        """Should split documents using default chunk size and overlap"""
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        documents = [mock_doc]

        mock_chunk = Mock()
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = [mock_chunk]
        mock_splitter_class.return_value = mock_splitter

        result = split_documents(documents)

        mock_splitter_class.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        assert result == [mock_chunk]

    @patch("src.ingestion.RecursiveCharacterTextSplitter")
    def test_splits_documents_with_custom_params(self, mock_splitter_class):
        """Should respect custom chunk size and overlap"""
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_class.return_value = mock_splitter

        split_documents([], chunk_size=500, chunk_overlap=100)

        mock_splitter_class.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @patch("src.ingestion.RecursiveCharacterTextSplitter")
    def test_returns_empty_list_for_empty_input(self, mock_splitter_class):
        """Should return empty list when given no documents"""
        mock_splitter = Mock()
        mock_splitter.split_documents.return_value = []
        mock_splitter_class.return_value = mock_splitter

        result = split_documents([])

        assert result == []

    def test_functional_split_documents(self):
        """Functional test using real splitter logic"""
        from langchain_core.documents import Document
        
        doc = Document(
            page_content="Chunk 1. Chunk 2. Chunk 3.",
            metadata={"source": "test.pdf"}
        )
        
        # Using very small chunk size to force splits at sentences
        result = split_documents([doc], chunk_size=10, chunk_overlap=0)
        
        assert len(result) >= 3
        assert result[0].metadata["source"] == "test.pdf"


class TestIngest:
    """Tests for ingest pipeline function"""

    @patch("src.ingestion.split_documents")
    @patch("src.ingestion.load_pdfs")
    def test_calls_load_and_split(self, mock_load, mock_split, tmp_path):
        """Should call load_pdfs then split_documents"""
        mock_docs = [Mock()]
        mock_chunks = [Mock(), Mock()]
        mock_load.return_value = mock_docs
        mock_split.return_value = mock_chunks

        result = ingest(str(tmp_path))

        mock_load.assert_called_once_with(str(tmp_path))
        mock_split.assert_called_once_with(mock_docs)
        assert result == mock_chunks

    @patch("src.ingestion.split_documents")
    @patch("src.ingestion.load_pdfs")
    def test_returns_empty_list_when_no_pdfs(self, mock_load, mock_split):
        """Should return empty list when no PDFs found"""
        mock_load.return_value = []
        mock_split.return_value = []

        result = ingest("./empty")

        assert result == []


class TestIngestionWithFixtures:
    """Integration tests using fixture documents"""

    def test_load_pdfs_from_fixtures(self):
        """Should load PDFs from fixtures directory"""
        docs = load_pdfs(str(FIXTURES_PATH))

        assert len(docs) >= 2
        # Check document structure
        assert hasattr(docs[0], 'page_content')
        assert hasattr(docs[0], 'metadata')

    def test_load_pdfs_content_from_fixtures(self):
        """Fixture PDFs should contain expected content"""
        docs = load_pdfs(str(FIXTURES_PATH))

        # Combine all content
        all_content = " ".join(doc.page_content for doc in docs)

        # Check for known facts from fixtures
        assert "revenue" in all_content.lower()
        assert "ceo" in all_content.lower()

    def test_ingest_full_pipeline_with_fixtures(self):
        """Full ingestion pipeline should work with fixtures"""
        chunks = ingest(str(FIXTURES_PATH))

        # Should produce chunks
        assert len(chunks) > 0

        # Each chunk should have content and metadata
        for chunk in chunks:
            assert hasattr(chunk, 'page_content')
            assert hasattr(chunk, 'metadata')
            assert len(chunk.page_content) > 0

    def test_ingest_preserves_source_metadata(self):
        """Chunks should preserve source file in metadata"""
        chunks = ingest(str(FIXTURES_PATH))

        # At least one chunk should have source metadata
        sources = [c.metadata.get('source', '') for c in chunks]
        assert any('.pdf' in s for s in sources)
