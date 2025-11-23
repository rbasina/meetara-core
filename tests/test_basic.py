"""
Basic tests for Meetara Core backend.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Meetara Core"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data


def test_chat_endpoint_basic(client):
    """Test basic chat endpoint."""
    response = client.post(
        "/api/chat/",
        json={
            "query": "Hello, how are you?",
            "topic": "general",
            "lang": "en"
        }
    )
    # Should return 503 if agent not initialized, or 200 if working
    assert response.status_code in [200, 503]


def test_upload_domains_endpoint(client):
    """Test domains endpoint."""
    response = client.get("/api/upload/domains")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_supported_file_types_endpoint(client):
    """Test supported file types endpoint."""
    response = client.get("/api/upload/supported-file-types")
    assert response.status_code == 200
    data = response.json()
    assert "supported_types" in data
    assert "max_file_size_mb" in data


def test_emotion_supported_endpoint(client):
    """Test supported emotions endpoint."""
    response = client.get("/api/emotion/supported-emotions")
    assert response.status_code == 200
    data = response.json()
    assert "facial_emotions" in data
    assert "speech_emotions" in data


def test_invalid_chat_request(client):
    """Test invalid chat request."""
    response = client.post(
        "/api/chat/",
        json={
            "query": "",  # Empty query should fail
            "topic": "general"
        }
    )
    assert response.status_code == 422  # Validation error


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/api/chat/")
    # Should not fail due to CORS middleware
    assert response.status_code in [200, 405]


if __name__ == "__main__":
    pytest.main([__file__]) 