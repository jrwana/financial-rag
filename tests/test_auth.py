import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# --- Query Auth (prod mode) ---

@patch("src.api.settings")
def test_query_prod_missing_key_returns_401(mock_settings):
    """Prod mode without API key should return 401"""
    mock_settings.ENV = "prod"
    mock_settings.API_KEY = "secret123"

    response = client.post("/query", json={"question": "Test?"})

    assert response.status_code == 401


@patch("src.api.settings")
def test_query_prod_wrong_key_returns_403(mock_settings):
    """Prod mode with wrong API key should return 403"""
    mock_settings.ENV = "prod"
    mock_settings.API_KEY = "secret123"

    response = client.post(
        "/query",
        json={"question": "Test?"},
        headers={"X-API-Key": "wrongkey"}
    )

    assert response.status_code == 403


@patch("app.chain", MagicMock())
@patch("app.query")
@patch("src.api.settings")
def test_query_prod_correct_key_succeeds(mock_settings, mock_query):
    """Prod mode with correct API key should succeed"""
    mock_settings.ENV = "prod"
    mock_settings.API_KEY = "secret123"
    mock_query.return_value = {"answer": "Answer", "sources": []}

    response = client.post(
        "/query",
        json={"question": "Test?"},
        headers={"X-API-Key": "secret123"}
    )

    assert response.status_code == 200


@patch("app.chain", MagicMock())
@patch("app.query")
@patch("src.api.settings")
def test_query_local_no_key_succeeds(mock_settings, mock_query):
    """Local mode should not require API key"""
    mock_settings.ENV = "local"
    mock_query.return_value = {"answer": "Answer", "sources": []}

    response = client.post("/query", json={"question": "Test?"})

    assert response.status_code == 200


# --- Ingest Auth (prod mode) ---

@patch("src.api.settings")
def test_ingest_prod_missing_key_returns_401(mock_settings):
    mock_settings.ENV = "prod"
    mock_settings.ADMIN_API_KEY = "admin123"

    response = client.post("/ingest")
    assert response.status_code == 401


@patch("src.api.settings")
def test_ingest_prod_wrong_key_returns_403(mock_settings):
    mock_settings.ENV = "prod"
    mock_settings.ADMIN_API_KEY = "admin123"

    response = client.post("/ingest", headers={"X-Admin-Key": "wrong"})
    assert response.status_code == 403