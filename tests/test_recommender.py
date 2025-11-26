"""
Tests for the Content Recommender
"""

import pytest
import json
import os
from app.recommender import ContentRecommender
from app.models import TierName, TIER_CONFIG


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def sample_content_data():
    """Sample content data for testing"""
    return {
        "version": "test-v1",
        "exportedAt": "2025-11-26T10:00:00Z",
        "vocabulary": [
            {"id": "v1", "hanzi": "你", "pinyin": "nǐ", "hskLevel": 1},
            {"id": "v2", "hanzi": "好", "pinyin": "hǎo", "hskLevel": 1},
            {"id": "v3", "hanzi": "你好", "pinyin": "nǐ hǎo", "hskLevel": 1},
            {"id": "v4", "hanzi": "我", "pinyin": "wǒ", "hskLevel": 1},
            {"id": "v5", "hanzi": "是", "pinyin": "shì", "hskLevel": 1},
            {"id": "v6", "hanzi": "学生", "pinyin": "xué shēng", "hskLevel": 1},
            {"id": "v7", "hanzi": "老师", "pinyin": "lǎo shī", "hskLevel": 1},
            {"id": "v8", "hanzi": "中国", "pinyin": "zhōng guó", "hskLevel": 2},
            {"id": "v9", "hanzi": "北京", "pinyin": "běi jīng", "hskLevel": 2},
            {"id": "v10", "hanzi": "旅行", "pinyin": "lǚ xíng", "hskLevel": 3},
        ],
        "lessons": [
            {"id": "lesson-1", "hskLevel": 1, "lessonNumber": 1, "title": "Greetings", "targetVocabulary": ["v1", "v2", "v3"]},
            {"id": "lesson-2", "hskLevel": 1, "lessonNumber": 2, "title": "Introduction", "targetVocabulary": ["v4", "v5", "v6"]},
            {"id": "lesson-3", "hskLevel": 1, "lessonNumber": 3, "title": "People", "targetVocabulary": ["v7"]},
            {"id": "lesson-4", "hskLevel": 2, "lessonNumber": 1, "title": "Places", "targetVocabulary": ["v8", "v9"]},
            {"id": "lesson-5", "hskLevel": 3, "lessonNumber": 1, "title": "Travel", "targetVocabulary": ["v10"]},
        ],
        "lessonOrder": ["lesson-1", "lesson-2", "lesson-3", "lesson-4", "lesson-5"],
        "lessonWordMap": {
            "lesson-1": ["v1", "v2", "v3"],
            "lesson-2": ["v4", "v5", "v6"],
            "lesson-3": ["v7"],
            "lesson-4": ["v8", "v9"],
            "lesson-5": ["v10"],
        },
        "stories": [
            {
                "id": "story-1",
                "title": "简单问候",
                "hskLevel": 1,
                "difficulty": "easy",
                "fullText": "你好！我是学生。",
                "sentenceCount": 2,
            },
            {
                "id": "story-2",
                "title": "去北京",
                "hskLevel": 2,
                "difficulty": "medium",
                "fullText": "我是学生。我去北京。北京是中国的。",
                "sentenceCount": 3,
            },
            {
                "id": "story-3",
                "title": "旅行故事",
                "hskLevel": 3,
                "difficulty": "hard",
                "fullText": "我去旅行。我去北京旅行。北京很好。",
                "sentenceCount": 3,
            },
        ],
        "audiobooks": [],
    }


@pytest.fixture
def recommender_with_data(tmp_path, sample_content_data):
    """Create a recommender with test data"""
    # Write content to temp file
    content_path = tmp_path / "content.json"
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(sample_content_data, f, ensure_ascii=False)
    
    recommender = ContentRecommender(data_dir=str(tmp_path))
    recommender.load()
    return recommender


# ═══════════════════════════════════════════════════════════
# Setup Tests
# ═══════════════════════════════════════════════════════════

class TestRecommenderSetup:
    def test_load_content(self, recommender_with_data):
        """Test loading content from file"""
        assert recommender_with_data.loaded is True
        assert len(recommender_with_data.vocab_by_id) == 10
        assert len(recommender_with_data.stories) == 3
    
    def test_cumulative_word_sets(self, recommender_with_data):
        """Test cumulative word sets are built correctly"""
        rec = recommender_with_data
        
        # Lesson 1 should have 3 words
        assert len(rec.cumulative_words["lesson-1"]) == 3
        assert "v1" in rec.cumulative_words["lesson-1"]
        
        # Lesson 2 should have 6 words (3 + 3)
        assert len(rec.cumulative_words["lesson-2"]) == 6
        
        # Lesson 3 should have 7 words
        assert len(rec.cumulative_words["lesson-3"]) == 7
        
        # Lesson 4 should have 9 words
        assert len(rec.cumulative_words["lesson-4"]) == 9
        
        # Lesson 5 should have all 10 words
        assert len(rec.cumulative_words["lesson-5"]) == 10
    
    def test_jieba_seeded_with_vocab(self, recommender_with_data):
        """Test jieba is seeded with vocabulary words"""
        import jieba
        
        # Multi-char words should be in jieba dict
        assert "你好" in jieba.dt.FREQ or len(list(jieba.cut("你好"))) == 1
    
    def test_get_info(self, recommender_with_data):
        """Test get_info returns correct structure"""
        info = recommender_with_data.get_info()
        
        assert info["loaded"] is True
        assert info["vocab_count"] == 10
        assert info["lesson_count"] == 5
        assert info["story_count"] == 3
        assert info["audiobook_count"] == 0


# ═══════════════════════════════════════════════════════════
# Comprehension Calculation Tests
# ═══════════════════════════════════════════════════════════

class TestComprehensionCalculation:
    def test_all_words_known(self, recommender_with_data):
        """Test comprehension with all known words"""
        rec = recommender_with_data
        
        # At lesson-3, user knows words from lessons 1-3
        known = rec.get_known_words_for_lesson("lesson-3")
        
        # Tokenize "你好我是学生" - all should be known
        tokens = rec._tokenize_content("你好我是学生")
        comp, unknown, count = rec._calculate_comprehension(tokens, known)
        
        # Should be 100% comprehension
        assert comp == 1.0
        assert count == 0
    
    def test_some_words_unknown(self, recommender_with_data):
        """Test comprehension with some unknown words"""
        rec = recommender_with_data
        
        # At lesson-2, user doesn't know 老师 (lesson-3)
        known = rec.get_known_words_for_lesson("lesson-2")
        
        tokens = rec._tokenize_content("我是老师")
        comp, unknown, count = rec._calculate_comprehension(tokens, known)
        
        # 老师 is unknown
        assert comp < 1.0
        assert count >= 1
    
    def test_empty_text(self, recommender_with_data):
        """Test comprehension with empty text"""
        rec = recommender_with_data
        known = rec.get_known_words_for_lesson("lesson-1")
        
        comp, unknown, count = rec._calculate_comprehension([], known)
        
        assert comp == 1.0
        assert count == 0


# ═══════════════════════════════════════════════════════════
# Recommendation Tests
# ═══════════════════════════════════════════════════════════

class TestRecommendations:
    def test_recommend_returns_tiers(self, recommender_with_data):
        """Test that recommendations return all tiers"""
        rec = recommender_with_data
        
        result = rec.recommend(lesson_id="lesson-3")
        
        assert "comfort" in result.tiers
        assert "challenge" in result.tiers
        assert "stretch" in result.tiers
    
    def test_tier_structure(self, recommender_with_data):
        """Test tier result structure"""
        rec = recommender_with_data
        
        result = rec.recommend(lesson_id="lesson-3")
        
        for tier_name, tier in result.tiers.items():
            assert hasattr(tier, "label")
            assert hasattr(tier, "description")
            assert hasattr(tier, "emoji")
            assert hasattr(tier, "range")
            assert hasattr(tier, "items")
    
    def test_recommend_content_type_filter(self, recommender_with_data):
        """Test filtering by content type"""
        rec = recommender_with_data
        
        # Stories only
        result = rec.recommend(lesson_id="lesson-3", content_type="story")
        
        for tier in result.tiers.values():
            for item in tier.items:
                assert item.type == "story"
    
    def test_recommend_items_per_tier_limit(self, recommender_with_data):
        """Test items per tier limit"""
        rec = recommender_with_data
        
        result = rec.recommend(lesson_id="lesson-3", items_per_tier=2)
        
        for tier in result.tiers.values():
            assert len(tier.items) <= 2
    
    def test_excluded_count(self, recommender_with_data):
        """Test excluded count for items below 75%"""
        rec = recommender_with_data
        
        # At lesson-1, user only knows 3 words
        # Stories with many unknown words should be excluded
        result = rec.recommend(lesson_id="lesson-1")
        
        # There might be excluded items
        assert isinstance(result.excludedCount, int)
    
    def test_unknown_words_preview(self, recommender_with_data):
        """Test unknown words are included in items"""
        rec = recommender_with_data
        
        result = rec.recommend(lesson_id="lesson-2")
        
        # Find an item with unknowns (if any)
        for tier in result.tiers.values():
            for item in tier.items:
                if item.unknownCount > 0:
                    assert len(item.unknownWords) > 0
                    assert item.unknownCount == len(item.unknownWords) or len(item.unknownWords) == 5


# ═══════════════════════════════════════════════════════════
# Tier Threshold Tests
# ═══════════════════════════════════════════════════════════

class TestTierThresholds:
    def test_comfort_tier_threshold(self):
        """Test comfort tier is 95-100%"""
        config = TIER_CONFIG[TierName.COMFORT]
        assert config["min"] == 0.95
        assert config["max"] == 1.00
    
    def test_challenge_tier_threshold(self):
        """Test challenge tier is 85-95%"""
        config = TIER_CONFIG[TierName.CHALLENGE]
        assert config["min"] == 0.85
        assert config["max"] == 0.95
    
    def test_stretch_tier_threshold(self):
        """Test stretch tier is 75-85%"""
        config = TIER_CONFIG[TierName.STRETCH]
        assert config["min"] == 0.75
        assert config["max"] == 0.85


# ═══════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_unknown_lesson_id(self, recommender_with_data):
        """Test with unknown lesson ID"""
        rec = recommender_with_data
        
        # Should return empty known words
        known = rec.get_known_words_for_lesson("unknown-lesson")
        assert len(known) == 0
    
    def test_recommend_with_no_content(self, tmp_path):
        """Test recommender with no stories"""
        data = {
            "version": "test",
            "exportedAt": "2025-11-26T10:00:00Z",
            "vocabulary": [],
            "lessons": [],
            "lessonOrder": [],
            "lessonWordMap": {},
            "stories": [],
            "audiobooks": [],
        }
        
        content_path = tmp_path / "content.json"
        with open(content_path, "w") as f:
            json.dump(data, f)
        
        rec = ContentRecommender(data_dir=str(tmp_path))
        rec.load()
        
        result = rec.recommend(lesson_id="any")
        
        # Should return empty tiers
        for tier in result.tiers.values():
            assert len(tier.items) == 0
    
    def test_punctuation_filtering(self, recommender_with_data):
        """Test punctuation is filtered from tokens"""
        rec = recommender_with_data
        
        tokens = rec._tokenize_content("你好！我是学生。")
        
        # Should not include punctuation
        for token in tokens:
            assert token["hanzi"] not in "！。，？"

