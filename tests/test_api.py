"""
API integration tests for vocab-validator FastAPI endpoints

Tests the HTTP API without external dependencies.
"""

import pytest
import json
import os
import tempfile
import shutil
from fastapi.testclient import TestClient


# We need to set up the data directory BEFORE importing the app
@pytest.fixture(scope="module")
def mock_data_dir():
    """Create mock data directory before app import"""
    temp_dir = tempfile.mkdtemp()
    
    # Create curriculum
    curriculum = {
        "words": {
            "你好": "hsk1-l1",
            "谢谢": "hsk1-l1",
            "再见": "hsk1-l1",
            "吃": "hsk1-l2",
            "喝": "hsk1-l2",
            "学习": "hsk1-l3",
            "可能": "hsk2-l1",
        },
        "version": "api-test-v1"
    }
    
    with open(os.path.join(temp_dir, "curriculum.json"), "w", encoding="utf-8") as f:
        json.dump(curriculum, f, ensure_ascii=False)
    
    with open(os.path.join(temp_dir, "version.txt"), "w") as f:
        f.write("api-test-v1")
    
    # Set env vars before import
    os.environ["DATA_DIR"] = temp_dir
    os.environ["ENVIRONMENT"] = "development"  # Allow no API key in tests
    os.environ["VALIDATOR_API_KEY"] = "test-api-key-12345"  # Set test API key
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def client(mock_data_dir):
    """Create test client with mock data"""
    # Import after setting DATA_DIR
    from app.main import app, validator
    
    # Force reload validator with test data
    validator.data_dir = mock_data_dir
    validator.load()
    
    return TestClient(app)


@pytest.fixture(scope="module")
def api_key():
    """Return the test API key"""
    return "test-api-key-12345"


class TestHealthEndpoint:
    """Tests for /health endpoint"""
    
    def test_health_returns_200(self, client):
        """Health check should return 200"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_shows_loaded(self, client):
        """Health should show curriculum is loaded"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["curriculum_loaded"] is True
        assert data["word_count"] > 0


class TestVersionEndpoint:
    """Tests for /version endpoint"""
    
    def test_version_returns_info(self, client):
        """Version endpoint should return curriculum info"""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "word_count" in data
        assert "loaded" in data


class TestValidateEndpoint:
    """Tests for /validate endpoint"""
    
    def test_validate_valid_text(self, client):
        """Should validate text with safe words"""
        response = client.post("/validate", json={
            "text": "你好！谢谢！",
            "user_position": {"hsk": 1, "lesson": 2},
            "target_words": []
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
    
    def test_validate_invalid_text(self, client):
        """Should reject text with forbidden words"""
        response = client.post("/validate", json={
            "text": "我可能去学习",  # 可能 is HSK2, 学习 is L3
            "user_position": {"hsk": 1, "lesson": 1},
            "target_words": []
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["forbidden_words"]) > 0
    
    def test_validate_with_target_words(self, client):
        """Should identify target words"""
        response = client.post("/validate", json={
            "text": "你好谢谢",
            "user_position": {"hsk": 1, "lesson": 1},
            "target_words": ["你好", "谢谢"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "你好" in data["target_words"] or "谢谢" in data["target_words"]
    
    def test_validate_returns_stats(self, client):
        """Should return word statistics"""
        response = client.post("/validate", json={
            "text": "你好再见",
            "user_position": {"hsk": 1, "lesson": 1},
            "target_words": []
        })
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "total_words" in data["stats"]
        assert "unique_words" in data["stats"]
    
    def test_validate_missing_text(self, client):
        """Should error on missing text"""
        response = client.post("/validate", json={
            "user_position": {"hsk": 1, "lesson": 1}
        })
        assert response.status_code == 422  # Validation error
    
    def test_validate_missing_position(self, client):
        """Should error on missing user position"""
        response = client.post("/validate", json={
            "text": "你好"
        })
        assert response.status_code == 422
    
    def test_validate_empty_text(self, client):
        """Should handle empty text"""
        response = client.post("/validate", json={
            "text": "",
            "user_position": {"hsk": 1, "lesson": 1},
            "target_words": []
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


class TestRequestValidation:
    """Tests for request validation"""
    
    def test_invalid_hsk_type(self, client):
        """Should reject non-integer HSK"""
        response = client.post("/validate", json={
            "text": "你好",
            "user_position": {"hsk": "one", "lesson": 1},
            "target_words": []
        })
        assert response.status_code == 422
    
    def test_invalid_lesson_type(self, client):
        """Should reject non-integer lesson"""
        response = client.post("/validate", json={
            "text": "你好",
            "user_position": {"hsk": 1, "lesson": "one"},
            "target_words": []
        })
        assert response.status_code == 422


class TestSyncEndpoint:
    """Tests for /sync endpoint security"""
    
    def test_sync_requires_api_key(self, client):
        """Should reject sync without API key"""
        response = client.post("/sync")
        assert response.status_code == 401
    
    def test_sync_rejects_wrong_key(self, client):
        """Should reject sync with wrong API key"""
        response = client.post("/sync", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401
    
    def test_sync_accepts_valid_key(self, client, api_key):
        """Should accept sync with valid API key"""
        # Note: This will fail to actually sync (no backend) but shouldn't 401
        response = client.post("/sync", headers={"X-API-Key": api_key})
        # Either 200 (success) or 500 (backend unreachable), but not 401
        assert response.status_code in [200, 500]


class TestHealthDetails:
    """Tests for health endpoint details"""
    
    def test_health_includes_environment(self, client):
        """Health should show environment"""
        response = client.get("/health")
        data = response.json()
        assert "environment" in data
        assert data["environment"] == "development"

