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
    
    # Create comprehensive curriculum for realistic tests
    # Words need to be added to jieba for proper segmentation
    curriculum = {
        "words": {
            # HSK 1, Lesson 1 - Basic greetings
            "你": "hsk1-l1",
            "好": "hsk1-l1",
            "你好": "hsk1-l1",
            "我": "hsk1-l1",
            "是": "hsk1-l1",
            "谢谢": "hsk1-l1",
            "再见": "hsk1-l1",
            "他": "hsk1-l1",
            "她": "hsk1-l1",
            # HSK 1, Lesson 2 - Actions
            "吃": "hsk1-l2",
            "喝": "hsk1-l2",
            "水": "hsk1-l2",
            "饭": "hsk1-l2",
            "看": "hsk1-l2",
            # HSK 1, Lesson 3 - Learning
            "学习": "hsk1-l3",
            "工作": "hsk1-l3",
            "学生": "hsk1-l3",
            "老师": "hsk1-l3",
            # HSK 2, Lesson 1 - Modal verbs
            "可能": "hsk2-l1",
            "应该": "hsk2-l1",
            "需要": "hsk2-l1",
            "能": "hsk2-l1",
            # HSK 2, Lesson 2 - Conjunctions
            "虽然": "hsk2-l2",
            "但是": "hsk2-l2",
            "因为": "hsk2-l2",
            "所以": "hsk2-l2",
            # HSK 3, Lesson 1 - Adjectives
            "聪明": "hsk3-l1",
            "努力": "hsk3-l1",
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
    import jieba
    
    # Import after setting DATA_DIR - this ensures the app reads our mock dir
    from app.main import app, validator
    
    # Clear jieba's dictionary and reload with fresh data
    validator.data_dir = mock_data_dir
    validator.curriculum = {}
    validator.loaded = False
    validator.load()
    
    # Re-add all curriculum words to jieba for proper segmentation
    for word in validator.curriculum.keys():
        jieba.add_word(word, freq=1000)  # High frequency to ensure it's used
    
    # Debug: verify curriculum loaded correctly
    assert len(validator.curriculum) > 20, f"Expected 20+ words, got {len(validator.curriculum)}"
    assert "可能" in validator.curriculum, "可能 should be in curriculum"
    assert validator.curriculum["可能"] == "hsk2-l1", f"可能 should be hsk2-l1, got {validator.curriculum['可能']}"
    
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
    
    # Note: Skipping test_sync_accepts_valid_key because it tries to sync 
    # from production backend and corrupts test curriculum data


class TestHealthDetails:
    """Tests for health endpoint details"""
    
    def test_health_includes_environment(self, client):
        """Health should show environment"""
        response = client.get("/health")
        data = response.json()
        assert "environment" in data
        assert data["environment"] == "development"


# ═══════════════════════════════════════════════════════════
# i+1 LESSON VALIDATION API TESTS
# ═══════════════════════════════════════════════════════════

class TestValidateLessonEndpoint:
    """Tests for /validate-lesson endpoint"""
    
    def test_validate_lesson_valid(self, client):
        """Should validate text with i+1 compliance"""
        response = client.post("/validate-lesson", json={
            "text": "你好！谢谢！",
            "lesson_number": 3,
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["invalid_words"]) == 0
    
    def test_validate_lesson_with_focus_words(self, client):
        """Should track focus words"""
        response = client.post("/validate-lesson", json={
            "text": "你好学习",
            "lesson_number": 3,
            "focus_words": ["学习"],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "学习" in data["focus_words_found"]
    
    def test_validate_lesson_invalid_words(self, client):
        """Should reject words from later lessons"""
        response = client.post("/validate-lesson", json={
            "text": "你好可能",  # 可能 is HSK2 L1
            "lesson_number": 3,  # HSK1 L3
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["invalid_words"]) > 0
    
    def test_validate_lesson_missing_focus_words(self, client):
        """Should fail when focus words missing"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "lesson_number": 3,
            "focus_words": ["学习"],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "学习" in data["focus_words_missing"]
        assert data["suggestion"] is not None
    
    def test_validate_lesson_returns_stats(self, client):
        """Should return statistics"""
        response = client.post("/validate-lesson", json={
            "text": "你好谢谢",
            "lesson_number": 3,
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "total_words" in data["stats"]
        assert "focus_coverage" in data["stats"]
    
    def test_validate_lesson_suggestion_on_fail(self, client):
        """Should provide suggestion on failure"""
        response = client.post("/validate-lesson", json={
            "text": "你好可能",
            "lesson_number": 3,
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["suggestion"] is not None


class TestValidateLessonRequestValidation:
    """Tests for request validation"""
    
    def test_missing_text(self, client):
        """Should error on missing text"""
        response = client.post("/validate-lesson", json={
            "lesson_number": 3,
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 422
    
    def test_missing_lesson_number(self, client):
        """Should error on missing lesson_number"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 422
    
    def test_missing_focus_words(self, client):
        """Should error on missing focus_words"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "lesson_number": 3,
            "hsk_level": 1
        })
        assert response.status_code == 422
    
    def test_default_hsk_level(self, client):
        """Should default to HSK 1 if not provided"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "lesson_number": 3,
            "focus_words": []
        })
        assert response.status_code == 200
    
    def test_invalid_lesson_number_type(self, client):
        """Should reject non-integer lesson_number"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "lesson_number": "three",
            "focus_words": [],
            "hsk_level": 1
        })
        assert response.status_code == 422
    
    def test_invalid_hsk_level_type(self, client):
        """Should reject non-integer hsk_level"""
        response = client.post("/validate-lesson", json={
            "text": "你好",
            "lesson_number": 3,
            "focus_words": [],
            "hsk_level": "one"
        })
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════
# GET VOCABULARY ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════

class TestGetVocabularyEndpoint:
    """Tests for /get-vocabulary endpoint - used by AI Tutor lesson generator"""
    
    def test_get_vocabulary_returns_200(self, client):
        """Should return 200 with vocabulary list"""
        response = client.get("/get-vocabulary?max_lesson=3")
        assert response.status_code == 200
    
    def test_get_vocabulary_returns_words(self, client):
        """Should return words list with count"""
        response = client.get("/get-vocabulary?max_lesson=3")
        data = response.json()
        assert "words" in data
        assert "count" in data
        assert "max_lesson" in data
        assert isinstance(data["words"], list)
        assert data["count"] == len(data["words"])
    
    def test_get_vocabulary_respects_max_lesson(self, client):
        """Should only return words up to max_lesson"""
        # Get words for lesson 1 (HSK1 L1 only)
        response1 = client.get("/get-vocabulary?max_lesson=1")
        data1 = response1.json()
        
        # Get words for lesson 3 (HSK1 L1-3)
        response3 = client.get("/get-vocabulary?max_lesson=3")
        data3 = response3.json()
        
        # Lesson 3 should have more words than lesson 1
        assert data3["count"] >= data1["count"]
        
        # Lesson 1 words should be subset of lesson 3 words
        for word in data1["words"]:
            assert word in data3["words"]
    
    def test_get_vocabulary_includes_basic_words(self, client):
        """Should include basic HSK1 L1 words"""
        response = client.get("/get-vocabulary?max_lesson=3")
        data = response.json()
        
        # These are HSK1 L1 words in our test curriculum
        basic_words = ["你", "好", "我", "是", "谢谢"]
        for word in basic_words:
            assert word in data["words"], f"{word} should be in vocabulary"
    
    def test_get_vocabulary_excludes_advanced_words(self, client):
        """Should NOT include words from later lessons"""
        response = client.get("/get-vocabulary?max_lesson=3")
        data = response.json()
        
        # These are HSK2+ words - should NOT be included for max_lesson=3
        advanced_words = ["可能", "应该", "虽然", "但是"]
        for word in advanced_words:
            assert word not in data["words"], f"{word} should NOT be in vocabulary for lesson 3"
    
    def test_get_vocabulary_default_max_lesson(self, client):
        """Should use default max_lesson=10 if not specified"""
        response = client.get("/get-vocabulary")
        assert response.status_code == 200
        data = response.json()
        assert data["max_lesson"] == 10
    
    def test_get_vocabulary_handles_high_lesson_number(self, client):
        """Should handle lesson numbers beyond curriculum"""
        response = client.get("/get-vocabulary?max_lesson=100")
        assert response.status_code == 200
        data = response.json()
        # Should return all available words
        assert data["count"] >= 0
    
    def test_get_vocabulary_handles_zero_lesson(self, client):
        """Should handle lesson 0 gracefully"""
        response = client.get("/get-vocabulary?max_lesson=0")
        assert response.status_code == 200
        data = response.json()
        # Should return empty or minimal set
        assert isinstance(data["words"], list)
    
    def test_get_vocabulary_handles_negative_lesson(self, client):
        """Should handle negative lesson number"""
        response = client.get("/get-vocabulary?max_lesson=-1")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["words"], list)


class TestValidateLessonIntegration:
    """Integration tests for the i+1 workflow"""
    
    def test_full_i1_workflow(self, client):
        """Test complete i+1 validation workflow"""
        # Simulate AI-generated text for lesson 3 with focus words 学习, 工作
        response = client.post("/validate-lesson", json={
            "text": "我学习。你工作。你好！",
            "lesson_number": 3,
            "focus_words": ["学习", "工作"],
            "hsk_level": 1
        })
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "学习" in data["focus_words_found"]
        assert "工作" in data["focus_words_found"]
        assert len(data["focus_words_missing"]) == 0
    
    def test_retry_with_feedback(self, client):
        """Test that feedback helps retry"""
        # First attempt - AI uses word from later lesson
        response1 = client.post("/validate-lesson", json={
            "text": "我可能学习",  # 可能 is HSK2
            "lesson_number": 3,
            "focus_words": ["学习"],
            "hsk_level": 1
        })
        data1 = response1.json()
        assert data1["valid"] is False
        # Should suggest replacing 可能
        assert "可能" in data1["suggestion"]
        
        # Second attempt - AI removes 可能
        response2 = client.post("/validate-lesson", json={
            "text": "我学习",  # Fixed version
            "lesson_number": 3,
            "focus_words": ["学习"],
            "hsk_level": 1
        })
        data2 = response2.json()
        assert data2["valid"] is True
    
    def test_hsk_progression(self, client):
        """Test that HSK level affects what's valid"""
        text = "我可能学习"  # 可能 is HSK2 L1
        
        # At HSK1 L3 - should fail (可能 is too advanced)
        response1 = client.post("/validate-lesson", json={
            "text": text,
            "lesson_number": 3,
            "focus_words": ["学习"],
            "hsk_level": 1
        })
        assert response1.json()["valid"] is False
        
        # At HSK2 L1 - should pass (可能 is current lesson focus)
        response2 = client.post("/validate-lesson", json={
            "text": text,
            "lesson_number": 1,
            "focus_words": ["可能"],
            "hsk_level": 2
        })
        assert response2.json()["valid"] is True

