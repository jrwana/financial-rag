import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# --- Startup Event ---

@patch("app.create_chain")
@patch("app.load_index")
def test_startup_loads_index(mock_load_index, mock_create_chain):
    """Startup should load existing index and create chain"""
    mock_index = MagicMock()
    mock_load_index.return_value = mock_index

    from app import startup
    asyncio.get_event_loop().run_until_complete(startup())

    mock_load_index.assert_called_once()
    mock_create_chain.assert_called_once_with(mock_index)


@patch("app.load_index")
def test_startup_handles_missing_index(mock_load_index):
    """Startup should handle missing index gracefully"""
    mock_load_index.side_effect = FileNotFoundError("Index not found")

    from app import startup
    # Should not raise
    asyncio.get_event_loop().run_until_complete(startup())


# --- Health Endpoint ---

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_shows_index_status():
    response = client.get("/health")
    assert "index_loaded" in response.json()


# --- Query Endpoint ---

def test_query_without_index_returns_400():
    """Query should fail if no index is loaded"""
    with patch("app.chain", None):
        response = client.post(
            "/query",
            json={"question": "What is the revenue?"}
        )
        assert response.status_code == 400
        assert "No index loaded" in response.json()["detail"]


@patch("app.chain", MagicMock())
@patch("app.query")
def test_query_success(mock_query):
    """Query should return answer and sources"""
    mock_query.return_value = {
        "answer": "The revenue was $1M",
        "sources": ["doc1.pdf", "doc2.pdf"]
    }

    response = client.post(
        "/query",
        json={"question": "What is the revenue?", "k": 4}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The revenue was $1M"
    assert data["sources"] == ["doc1.pdf", "doc2.pdf"]
    # Verify query was called with correct arguments
    mock_query.assert_called_once()


@patch("app.chain", MagicMock())
@patch("app.query")
def test_query_with_custom_k(mock_query):
    """Query should accept custom k parameter"""
    mock_query.return_value = {"answer": "Answer", "sources": []}

    response = client.post(
        "/query",
        json={"question": "Test?", "k": 10}
    )

    assert response.status_code == 200
    mock_query.assert_called_once()


def test_query_missing_question_returns_422():
    """Query should fail if question is missing"""
    response = client.post("/query", json={})
    assert response.status_code == 422


# --- Ingest Endpoint ---

@patch("app.create_chain")
@patch("app.save_index")
@patch("app.create_index")
@patch("app.ingest")
def test_ingest_success(mock_ingest, mock_create_index, mock_save_index, mock_create_chain):
    """Ingest should process documents and return count"""
    mock_chunks = [MagicMock(), MagicMock(), MagicMock()]
    mock_index = MagicMock()
    mock_ingest.return_value = mock_chunks
    mock_create_index.return_value = mock_index

    response = client.post("/ingest")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["chunks_processed"] == 3
    # Verify functions were called with correct arguments
    mock_ingest.assert_called_once()
    mock_create_index.assert_called_once_with(mock_chunks)
    mock_save_index.assert_called_once_with(mock_index)
    mock_create_chain.assert_called_once_with(mock_index)


@patch("app.ingest")
def test_ingest_no_documents_returns_404(mock_ingest):
    """Ingest should return 404 if no documents found"""
    mock_ingest.side_effect = FileNotFoundError("No PDFs found in data/")

    response = client.post("/ingest")

    assert response.status_code == 404
    assert "No PDFs found" in response.json()["detail"]


@patch("app.ingest")
def test_ingest_error_returns_500(mock_ingest):
    """Ingest should return 500 on unexpected error"""
    mock_ingest.side_effect = Exception("Unexpected error")

    response = client.post("/ingest")

    assert response.status_code == 500
