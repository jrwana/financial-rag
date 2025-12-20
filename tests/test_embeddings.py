"""Unit tests for embeddings module"""

import pytest
from unittest.mock import Mock, patch, ANY
from langchain_community.vectorstores import FAISS

from src.embeddings import (
    get_embeddings,
    create_index,
    save_index,
    load_index,
    DEFAULT_MODEL,
    INDEX_DIR,
)


class TestGetEmbeddings:
    """Tests for get_embeddings function"""

    @patch("src.embeddings.HuggingFaceEmbeddings")
    def test_initializes_default_model(self, mock_hf_embeddings):
        """Should initialize HuggingFaceEmbeddings with default model"""
        get_embeddings()
        
        mock_hf_embeddings.assert_called_once_with(model_name=DEFAULT_MODEL)

    @patch("src.embeddings.HuggingFaceEmbeddings")
    def test_initializes_custom_model(self, mock_hf_embeddings):
        """Should initialize HuggingFaceEmbeddings with custom model"""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        get_embeddings(custom_model)
        
        mock_hf_embeddings.assert_called_once_with(model_name=custom_model)


class TestCreateIndex:
    """Tests for create_index function"""

    @patch("src.embeddings.get_embeddings")
    @patch("src.embeddings.FAISS")
    def test_creates_index_from_documents(self, mock_faiss, mock_get_embeddings):
        """Should create FAISS index from document chunks"""
        mock_chunks = [Mock(), Mock()]
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_index = Mock()
        mock_faiss.from_documents.return_value = mock_index

        result = create_index(mock_chunks)

        mock_get_embeddings.assert_called_once_with(DEFAULT_MODEL)
        mock_faiss.from_documents.assert_called_once_with(mock_chunks, mock_embeddings)
        assert result == mock_index

    @patch("src.embeddings.get_embeddings")
    @patch("src.embeddings.FAISS")
    def test_creates_index_with_custom_model(self, mock_faiss, mock_get_embeddings):
        """Should use custom model name when creating index"""
        mock_chunks = []
        custom_model = "custom/model"
        
        create_index(mock_chunks, model_name=custom_model)

        mock_get_embeddings.assert_called_once_with(custom_model)


class TestSaveIndex:
    """Tests for save_index function"""

    @patch("pathlib.Path.mkdir")
    def test_saves_index_locally(self, mock_mkdir):
        """Should save FAISS index to disk"""
        mock_index = Mock(spec=FAISS)
        
        save_index(mock_index)

        # Check directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Check save call
        mock_index.save_local.assert_called_once_with(INDEX_DIR)

    @patch("pathlib.Path.mkdir")
    def test_saves_index_to_custom_dir(self, mock_mkdir):
        """Should save FAISS index to specified directory"""
        mock_index = Mock(spec=FAISS)
        custom_dir = "./custom_index"
        
        save_index(mock_index, index_dir=custom_dir)

        mock_index.save_local.assert_called_once_with(custom_dir)


class TestLoadIndex:
    """Tests for load_index function"""

    @patch("src.embeddings.Path")
    @patch("src.embeddings.get_embeddings")
    @patch("src.embeddings.FAISS")
    def test_loads_index_from_disk(self, mock_faiss, mock_get_embeddings, mock_path):
        """Should load FAISS index from disk with allow_dangerous_deserialization=True"""
        mock_path.return_value.exists.return_value = True
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_index = Mock()
        mock_faiss.load_local.return_value = mock_index

        result = load_index()

        mock_get_embeddings.assert_called_once_with(DEFAULT_MODEL)
        mock_faiss.load_local.assert_called_once_with(
            INDEX_DIR,
            mock_embeddings,
            allow_dangerous_deserialization=True
        )
        assert result == mock_index

    @patch("src.embeddings.Path")
    @patch("src.embeddings.get_embeddings")
    @patch("src.embeddings.FAISS")
    def test_loads_index_with_custom_params(self, mock_faiss, mock_get_embeddings, mock_path):
        """Should load index using custom directory and model"""
        mock_path.return_value.exists.return_value = True
        custom_dir = "./my_index"
        custom_model = "my/model"

        load_index(index_dir=custom_dir, model_name=custom_model)

        mock_get_embeddings.assert_called_once_with(custom_model)
        mock_faiss.load_local.assert_called_once_with(
            custom_dir,
            ANY,
            allow_dangerous_deserialization=True
        )

    @patch("src.embeddings.Path")
    def test_raises_error_when_index_missing(self, mock_path):
        """Should raise FileNotFoundError when index directory doesn't exist"""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError):
            load_index()
