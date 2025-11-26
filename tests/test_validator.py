"""
Unit tests for VocabValidator

Tests the core validation logic without network calls.
"""

import pytest
from app.validator import VocabValidator


class TestValidatorSetup:
    """Tests for validator initialization and loading"""
    
    def test_load_curriculum(self, validator):
        """Should load curriculum successfully"""
        assert validator.loaded is True
        assert validator.version == "test-v1"
        assert len(validator.curriculum) > 0
    
    def test_get_curriculum_info(self, validator):
        """Should return curriculum info"""
        info = validator.get_curriculum_info()
        assert info["loaded"] is True
        assert info["word_count"] == 15  # Number of words in mock
        assert info["version"] == "test-v1"
    
    def test_missing_curriculum_raises(self):
        """Should raise error if curriculum doesn't exist"""
        v = VocabValidator(data_dir="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            v.load()


class TestPositionParsing:
    """Tests for position string parsing"""
    
    def test_parse_hsk1_l1(self, validator):
        """Should parse hsk1-l1 correctly"""
        hsk, lesson = validator._parse_position("hsk1-l1")
        assert hsk == 1
        assert lesson == 1
    
    def test_parse_hsk2_l5(self, validator):
        """Should parse hsk2-l5 correctly"""
        hsk, lesson = validator._parse_position("hsk2-l5")
        assert hsk == 2
        assert lesson == 5
    
    def test_parse_invalid_returns_zeros(self, validator):
        """Should return (0,0) for invalid position"""
        hsk, lesson = validator._parse_position("invalid")
        assert hsk == 0
        assert lesson == 0


class TestWordSafety:
    """Tests for determining if words are safe for user's level"""
    
    def test_always_safe_words(self, validator):
        """Common words like 我, 你, 是 should always be safe"""
        assert validator._is_word_safe("我", 1, 1) is True
        assert validator._is_word_safe("你", 1, 1) is True
        assert validator._is_word_safe("是", 1, 1) is True
        assert validator._is_word_safe("不", 1, 1) is True
    
    def test_word_from_earlier_lesson_is_safe(self, validator):
        """Words from earlier lessons should be safe"""
        # User is at HSK1 L3, 你好 is from HSK1 L1
        assert validator._is_word_safe("你好", 1, 3) is True
    
    def test_word_from_same_lesson_is_safe(self, validator):
        """Words from same lesson should be safe"""
        # User is at HSK1 L1, 你好 is from HSK1 L1
        assert validator._is_word_safe("你好", 1, 1) is True
    
    def test_word_from_later_lesson_not_safe(self, validator):
        """Words from later lessons should not be safe"""
        # User is at HSK1 L1, 学习 is from HSK1 L3
        assert validator._is_word_safe("学习", 1, 1) is False
    
    def test_word_from_earlier_hsk_is_safe(self, validator):
        """Words from earlier HSK levels should be safe"""
        # User is at HSK2 L1, 你好 is from HSK1 L1
        assert validator._is_word_safe("你好", 2, 1) is True
    
    def test_word_from_later_hsk_not_safe(self, validator):
        """Words from later HSK levels should not be safe"""
        # User is at HSK1 L3, 可能 is from HSK2 L1
        assert validator._is_word_safe("可能", 1, 3) is False
    
    def test_unknown_word_not_safe(self, validator):
        """Words not in curriculum should not be safe"""
        assert validator._is_word_safe("随便", 1, 1) is False


class TestTargetWord:
    """Tests for identifying target words"""
    
    def test_current_lesson_word_is_target(self, validator):
        """Word from current lesson should be target"""
        # User at HSK1 L1, 你好 from HSK1 L1
        assert validator._is_target_word("你好", 1, 1) is True
    
    def test_earlier_lesson_word_not_target(self, validator):
        """Word from earlier lesson should not be target"""
        # User at HSK1 L2, 你好 from HSK1 L1
        assert validator._is_target_word("你好", 1, 2) is False
    
    def test_later_lesson_word_not_target(self, validator):
        """Word from later lesson should not be target"""
        # User at HSK1 L1, 学习 from HSK1 L3
        assert validator._is_target_word("学习", 1, 1) is False


class TestValidation:
    """Tests for full validation flow"""
    
    def test_valid_text_with_safe_words_only(self, validator):
        """Text with only safe words should be valid"""
        # User at HSK1 L2, using words from L1
        result = validator.validate(
            text="你好！谢谢！",
            user_hsk=1,
            user_lesson=2
        )
        assert result["valid"] is True
        assert len(result["forbidden_words"]) == 0
    
    def test_invalid_text_with_forbidden_words(self, validator):
        """Text with forbidden words should be invalid"""
        # User at HSK1 L1, using word from HSK2
        result = validator.validate(
            text="我可能去。",  # 可能 is HSK2
            user_hsk=1,
            user_lesson=1
        )
        assert result["valid"] is False
        assert "可能" in result["forbidden_words"]
    
    def test_target_words_identified(self, validator):
        """Target words should be identified correctly"""
        result = validator.validate(
            text="你好，谢谢。",
            user_hsk=1,
            user_lesson=1,
            target_words=["你好", "谢谢"]
        )
        assert result["valid"] is True
        assert "你好" in result["target_words"]
        assert "谢谢" in result["target_words"]
    
    def test_unknown_words_identified(self, validator):
        """Words not in curriculum should be marked unknown"""
        result = validator.validate(
            text="这是披萨。",  # 披萨 not in curriculum
            user_hsk=1,
            user_lesson=1
        )
        # Unknown words don't fail validation
        assert "披萨" in result["unknown_words"]
    
    def test_punctuation_filtered(self, validator):
        """Punctuation should be filtered out"""
        result = validator.validate(
            text="你好！！！",
            user_hsk=1,
            user_lesson=1
        )
        assert "！" not in result["words_found"]
    
    def test_stats_calculated(self, validator):
        """Stats should be calculated correctly"""
        result = validator.validate(
            text="你好谢谢再见",
            user_hsk=1,
            user_lesson=1
        )
        assert result["stats"]["total_words"] > 0
        assert result["stats"]["unique_words"] > 0
        assert "safe_percentage" in result["stats"]
    
    def test_always_safe_in_all_contexts(self, validator):
        """Function words should be safe even at earliest level"""
        result = validator.validate(
            text="我是你的",  # All function words
            user_hsk=1,
            user_lesson=1
        )
        assert result["valid"] is True
        assert len(result["forbidden_words"]) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_empty_text(self, validator):
        """Empty text should validate"""
        result = validator.validate(
            text="",
            user_hsk=1,
            user_lesson=1
        )
        assert result["valid"] is True
        assert result["stats"]["total_words"] == 0
    
    def test_only_punctuation(self, validator):
        """Text with only punctuation should be valid"""
        result = validator.validate(
            text="！？。，",
            user_hsk=1,
            user_lesson=1
        )
        assert result["valid"] is True
    
    def test_mixed_content(self, validator):
        """Mixed safe, target, and unknown words"""
        result = validator.validate(
            text="我学习吃披萨",  # 我=safe, 学习=target, 吃=forbidden for L1, 披萨=unknown
            user_hsk=1,
            user_lesson=3,
            target_words=["学习"]
        )
        # At HSK1 L3, 吃 (from L2) should be safe, 学习 (from L3) is target
        assert result["valid"] is True
        assert "学习" in result["target_words"]
        assert "披萨" in result["unknown_words"]
    
    def test_user_at_hsk3_sees_all_earlier_as_safe(self, validator):
        """User at HSK3 should see all HSK1-2 words as safe"""
        result = validator.validate(
            text="你好可能虽然",  # Words from HSK1 and HSK2
            user_hsk=3,
            user_lesson=1
        )
        assert result["valid"] is True
        assert len(result["forbidden_words"]) == 0


# ═══════════════════════════════════════════════════════════
# i+1 LESSON VALIDATION TESTS
# ═══════════════════════════════════════════════════════════

class TestValidateLessonBasic:
    """Basic tests for i+1 lesson validation"""
    
    def test_valid_text_with_focus_words(self, validator):
        """Text with only known words and focus words should pass"""
        result = validator.validate_lesson(
            text="你好！学习很好！",  # 你好=L1, 学习=L3 (focus)
            lesson_number=3,
            focus_words=["学习"],
            hsk_level=1
        )
        assert result["valid"] is True
        assert "学习" in result["focus_words_found"]
        assert len(result["invalid_words"]) == 0
    
    def test_invalid_word_from_later_lesson(self, validator):
        """Words from later lessons should fail"""
        result = validator.validate_lesson(
            text="你好可能",  # 可能 is HSK2 L1 (lesson 11)
            lesson_number=3,  # HSK1 L3 (lesson 3)
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is False
        assert len(result["invalid_words"]) > 0
        # Check that 可能 is in invalid words
        invalid_word_names = [w["word"] for w in result["invalid_words"]]
        assert "可能" in invalid_word_names
    
    def test_focus_word_exempts_from_lesson_check(self, validator):
        """Focus words should be allowed even if from later"""
        result = validator.validate_lesson(
            text="你好可能",  # 可能 is HSK2 L1
            lesson_number=3,
            focus_words=["可能"],  # Explicitly allowed as focus
            hsk_level=1
        )
        assert result["valid"] is True
        assert "可能" in result["focus_words_found"]
    
    def test_missing_focus_words_fails(self, validator):
        """Text without all focus words should fail"""
        result = validator.validate_lesson(
            text="你好",  # Missing 学习
            lesson_number=3,
            focus_words=["学习"],
            hsk_level=1
        )
        assert result["valid"] is False
        assert "学习" in result["focus_words_missing"]
        assert result["suggestion"] is not None


class TestValidateLessonAbsolutePosition:
    """Tests for absolute lesson position calculation"""
    
    def test_hsk1_lesson_1(self, validator):
        """HSK1 L1 = absolute lesson 1"""
        result = validator.validate_lesson(
            text="你好",  # HSK1 L1
            lesson_number=1,
            focus_words=["你好"],
            hsk_level=1
        )
        assert result["valid"] is True
    
    def test_hsk1_lesson_3(self, validator):
        """HSK1 L3 = absolute lesson 3, L1-L2 words should be safe"""
        result = validator.validate_lesson(
            text="你好吃喝",  # L1: 你好, L2: 吃,喝
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True
    
    def test_hsk2_can_use_all_hsk1(self, validator):
        """HSK2 L1 = absolute lesson 11, all HSK1 should be safe"""
        result = validator.validate_lesson(
            text="你好学习工作",  # All from HSK1
            lesson_number=1,
            focus_words=[],
            hsk_level=2
        )
        assert result["valid"] is True
    
    def test_hsk2_lesson_2_allows_hsk2_lesson_1(self, validator):
        """HSK2 L2 should allow HSK2 L1 words"""
        result = validator.validate_lesson(
            text="可能应该",  # HSK2 L1 words
            lesson_number=2,
            focus_words=[],
            hsk_level=2
        )
        assert result["valid"] is True


class TestValidateLessonFocusWords:
    """Tests for focus word handling"""
    
    def test_all_focus_words_found(self, validator):
        """Should track which focus words are found"""
        result = validator.validate_lesson(
            text="你好学习工作",
            lesson_number=3,
            focus_words=["学习", "工作"],
            hsk_level=1
        )
        assert result["valid"] is True
        assert "学习" in result["focus_words_found"]
        assert "工作" in result["focus_words_found"]
        assert len(result["focus_words_missing"]) == 0
    
    def test_partial_focus_words(self, validator):
        """Should identify missing focus words"""
        result = validator.validate_lesson(
            text="你好学习",  # Missing 工作
            lesson_number=3,
            focus_words=["学习", "工作"],
            hsk_level=1
        )
        assert result["valid"] is False
        assert "学习" in result["focus_words_found"]
        assert "工作" in result["focus_words_missing"]
    
    def test_no_focus_words_required(self, validator):
        """Should pass if no focus words specified"""
        result = validator.validate_lesson(
            text="你好",
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True


class TestValidateLessonAlwaysSafe:
    """Tests for always-safe words"""
    
    def test_pronouns_always_safe(self, validator):
        """Pronouns should be allowed at any level"""
        result = validator.validate_lesson(
            text="我你他她它",  # All pronouns
            lesson_number=1,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True
    
    def test_particles_always_safe(self, validator):
        """Particles should be allowed"""
        result = validator.validate_lesson(
            text="是的吗呢了",
            lesson_number=1,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True


class TestValidateLessonUnknownWords:
    """Tests for unknown word handling"""
    
    def test_unknown_words_tracked(self, validator):
        """Words not in curriculum should be tracked"""
        result = validator.validate_lesson(
            text="你好披萨",  # 披萨 not in curriculum
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert "披萨" in result["unknown_words"]
    
    def test_unknown_words_dont_fail(self, validator):
        """Unknown words shouldn't cause failure (might be valid Chinese)"""
        result = validator.validate_lesson(
            text="你好",
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True


class TestValidateLessonSuggestions:
    """Tests for suggestion generation"""
    
    def test_suggestion_for_invalid_words(self, validator):
        """Should provide suggestion when words are too advanced"""
        result = validator.validate_lesson(
            text="你好可能虽然",  # 可能, 虽然 are HSK2
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is False
        assert result["suggestion"] is not None
        assert "可能" in result["suggestion"] or "虽然" in result["suggestion"]
    
    def test_suggestion_for_missing_focus(self, validator):
        """Should suggest adding missing focus words"""
        result = validator.validate_lesson(
            text="你好",
            lesson_number=3,
            focus_words=["学习"],
            hsk_level=1
        )
        assert result["valid"] is False
        assert "学习" in result["suggestion"]


class TestValidateLessonStats:
    """Tests for statistics calculation"""
    
    def test_stats_returned(self, validator):
        """Should return word statistics"""
        result = validator.validate_lesson(
            text="你好你好学习",  # Repeated 你好
            lesson_number=3,
            focus_words=["学习"],
            hsk_level=1
        )
        assert "stats" in result
        assert result["stats"]["total_words"] >= 2
        assert result["stats"]["unique_words"] >= 2
        assert result["stats"]["focus_coverage"] == "1/1"
    
    def test_focus_coverage_format(self, validator):
        """Focus coverage should be 'found/total' format"""
        result = validator.validate_lesson(
            text="你好学习",
            lesson_number=3,
            focus_words=["学习", "工作"],
            hsk_level=1
        )
        assert result["stats"]["focus_coverage"] == "1/2"


class TestValidateLessonEdgeCases:
    """Edge cases for lesson validation"""
    
    def test_empty_text(self, validator):
        """Empty text should fail if focus words required"""
        result = validator.validate_lesson(
            text="",
            lesson_number=3,
            focus_words=["学习"],
            hsk_level=1
        )
        assert result["valid"] is False
        assert "学习" in result["focus_words_missing"]
    
    def test_empty_text_no_focus(self, validator):
        """Empty text with no focus words should pass"""
        result = validator.validate_lesson(
            text="",
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True
    
    def test_punctuation_ignored(self, validator):
        """Punctuation should not be counted"""
        result = validator.validate_lesson(
            text="你好！！！",
            lesson_number=3,
            focus_words=[],
            hsk_level=1
        )
        assert result["valid"] is True
        # Punctuation shouldn't be in any list
        for word in result.get("invalid_words", []):
            assert word["word"] != "！"

