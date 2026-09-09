import pytest
from fastapi.testclient import TestClient
from backend.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "database" in data
        assert "timestamp" in data
        assert data["service"] == "Interview Copilot AI Backend"
