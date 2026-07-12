"""Tests for FastAPI backend endpoints."""

from fastapi.testclient import TestClient


class TestAPIHealth:
    """Tests for the health endpoint."""

    def test_health_endpoint(self):
        """Health endpoint should return 200."""
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_models_endpoint(self):
        """Models endpoint should list available models."""
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

    def test_analyses_endpoint(self):
        """Analyses endpoint should return list."""
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/api/analyses")

        assert response.status_code == 200
        data = response.json()
        assert "analyses" in data


class TestAPISchemas:
    """Tests for request/response schemas."""

    def test_sample_record_request_defaults(self):
        """SampleRecordRequest should have correct defaults."""
        from src.api.schemas import SampleRecordRequest

        req = SampleRecordRequest()
        assert req.record_name == "100"
        assert req.model_name == "cnn_baseline"

    def test_sample_record_request_custom(self):
        """SampleRecordRequest should accept custom values."""
        from src.api.schemas import SampleRecordRequest

        req = SampleRecordRequest(record_name="200", model_name="cnn_lstm")
        assert req.record_name == "200"
        assert req.model_name == "cnn_lstm"

    def test_aami_class_enum(self):
        """AAMIClass should have all 5 superclasses."""
        from src.api.schemas import AAMIClass

        classes = [AAMIClass.N, AAMIClass.S, AAMIClass.V, AAMIClass.F, AAMIClass.Q]
        assert len(classes) == 5
