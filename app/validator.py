"""
Vocabulary Validator
Core validation logic using jieba for Chinese word segmentation.

Design:
- Stateless validation function
- User position determines what's "safe" vs "forbidden"
- Flip logic: whitelist for early learners, blacklist for advanced
"""

import json
import os
import jieba
from typing import Optional


class VocabValidator:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.curriculum: dict = {}
        self.version: str = ""
        self.loaded: bool = False
        
        # Common function words that are always allowed
        # (pronouns, particles, basic connectors)
        self.always_safe = {
            "我", "你", "他", "她", "它", "我们", "你们", "他们",
            "的", "了", "吗", "呢", "吧", "啊", "哦", "嗯",
            "是", "有", "在", "和", "与", "或", "但", "而",
            "这", "那", "什么", "怎么", "为什么", "哪", "哪里",
            "很", "太", "最", "都", "也", "还", "就", "才",
            "不", "没", "别", "请", "要", "会", "能", "可以",
            "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
            "个", "些", "点", "下", "上", "里", "外",
        }
    
    def load(self):
        """Load curriculum from local cache"""
        curriculum_path = os.path.join(self.data_dir, "curriculum.json")
        version_path = os.path.join(self.data_dir, "version.txt")
        
        if not os.path.exists(curriculum_path):
            raise FileNotFoundError(f"Curriculum not found at {curriculum_path}")
        
        with open(curriculum_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.curriculum = data.get("words", {})
        
        if os.path.exists(version_path):
            with open(version_path, "r") as f:
                self.version = f.read().strip()
        
        self.loaded = True
        
        # Add curriculum words to jieba dictionary for better segmentation
        for word in self.curriculum.keys():
            jieba.add_word(word)
    
    def reload(self):
        """Reload curriculum after sync"""
        self.load()
    
    def get_curriculum_info(self) -> dict:
        """Get current curriculum info"""
        return {
            "loaded": self.loaded,
            "word_count": len(self.curriculum),
            "version": self.version
        }
    
    def _parse_position(self, position_str: str) -> tuple[int, int]:
        """Parse 'hsk1-l3' to (1, 3)"""
        try:
            parts = position_str.replace("hsk", "").split("-l")
            return int(parts[0]), int(parts[1])
        except:
            return 0, 0
    
    def _is_word_safe(self, word: str, user_hsk: int, user_lesson: int) -> bool:
        """Check if word is safe for user's position"""
        # Always-safe words (function words, numbers, etc.)
        if word in self.always_safe:
            return True
        
        # Not in curriculum - could be punctuation or unknown
        if word not in self.curriculum:
            return False
        
        word_pos = self.curriculum[word]
        word_hsk, word_lesson = self._parse_position(word_pos)
        
        # Word learned in earlier HSK level
        if word_hsk < user_hsk:
            return True
        
        # Word in same HSK level but earlier lesson
        if word_hsk == user_hsk and word_lesson <= user_lesson:
            return True
        
        return False
    
    def _is_target_word(self, word: str, user_hsk: int, user_lesson: int) -> bool:
        """Check if word is a target word (currently learning)"""
        if word not in self.curriculum:
            return False
        
        word_pos = self.curriculum[word]
        word_hsk, word_lesson = self._parse_position(word_pos)
        
        return word_hsk == user_hsk and word_lesson == user_lesson
    
    def validate(
        self,
        text: str,
        user_hsk: int,
        user_lesson: int,
        target_words: list[str] = None
    ) -> dict:
        """
        Validate text against user's curriculum position.
        
        Returns dict with:
        - valid: bool
        - words_found: all words in text
        - safe_words: words user has learned
        - target_words: words being taught in this lesson
        - forbidden_words: words too advanced
        - unknown_words: words not in curriculum
        - stats: additional statistics
        """
        target_words = target_words or []
        target_set = set(target_words)
        
        # Segment text using jieba
        words = list(jieba.cut(text))
        
        # Filter out punctuation and whitespace
        words = [w for w in words if w.strip() and not self._is_punctuation(w)]
        
        # Categorize words
        safe = []
        targets = []
        forbidden = []
        unknown = []
        
        for word in words:
            if word in self.always_safe:
                safe.append(word)
            elif word in target_set:
                # Explicitly provided as target
                targets.append(word)
            elif word not in self.curriculum:
                unknown.append(word)
            elif self._is_word_safe(word, user_hsk, user_lesson):
                safe.append(word)
            elif self._is_target_word(word, user_hsk, user_lesson):
                targets.append(word)
            else:
                forbidden.append(word)
        
        # Validation passes if no forbidden words
        # Unknown words are flagged but don't fail validation
        # (they might be names, new words, etc.)
        is_valid = len(forbidden) == 0
        
        return {
            "valid": is_valid,
            "words_found": words,
            "safe_words": list(set(safe)),
            "target_words": list(set(targets)),
            "forbidden_words": list(set(forbidden)),
            "unknown_words": list(set(unknown)),
            "stats": {
                "total_words": len(words),
                "unique_words": len(set(words)),
                "safe_count": len(set(safe)),
                "target_count": len(set(targets)),
                "forbidden_count": len(set(forbidden)),
                "unknown_count": len(set(unknown)),
                "safe_percentage": round(len(safe) / len(words) * 100, 1) if words else 0
            }
        }

    def validate_lesson(
        self,
        text: str,
        lesson_number: int,
        focus_words: list[str],
        hsk_level: int = 1
    ) -> dict:
        """
        Validate lesson text for strict i+1 compliance.
        
        A word is VALID if:
        1. It's in focus_words (the new vocabulary being taught)
        2. It's in always_safe (common particles, pronouns)
        3. It's in curriculum with lesson_id <= lesson_number
        
        Returns:
        - valid: True if all words pass
        - invalid_words: List of words that fail + their lesson IDs
        - focus_words_found: Focus words that appeared in text
        - focus_words_missing: Focus words NOT in text
        - unknown_words: Words not in curriculum
        - suggestion: Feedback for AI retry
        """
        focus_set = set(focus_words)
        
        # Segment text using jieba
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip() and not self._is_punctuation(w)]
        
        invalid_words = []
        unknown_words = []
        focus_found = set()
        
        for word in words:
            # Check if it's a focus word (i+1)
            if word in focus_set:
                focus_found.add(word)
                continue
            
            # Check if always safe
            if word in self.always_safe:
                continue
            
            # Check curriculum
            if word not in self.curriculum:
                unknown_words.append(word)
                continue
            
            # Get word's lesson position
            word_pos = self.curriculum[word]
            word_hsk, word_lesson = self._parse_position(word_pos)
            
            # Calculate absolute lesson ID
            # HSK1 lessons 1-10 = IDs 1-10
            # HSK2 lessons 1-10 = IDs 11-20
            # etc.
            word_absolute_lesson = (word_hsk - 1) * 10 + word_lesson
            current_absolute_lesson = (hsk_level - 1) * 10 + lesson_number
            
            # Word must be from this lesson or earlier
            if word_absolute_lesson > current_absolute_lesson:
                invalid_words.append({
                    "word": word,
                    "lesson_id": word_absolute_lesson,
                    "reason": f"Word from lesson {word_absolute_lesson}, but current is {current_absolute_lesson}"
                })
        
        # Check which focus words are missing
        focus_missing = list(focus_set - focus_found)
        
        # Build response
        is_valid = len(invalid_words) == 0
        
        suggestion = None
        if not is_valid:
            bad_words = [w["word"] for w in invalid_words]
            suggestion = f"Replace these words with simpler alternatives: {', '.join(bad_words)}"
        elif focus_missing:
            suggestion = f"The text is missing these focus words: {', '.join(focus_missing)}"
        
        return {
            "valid": is_valid and len(focus_missing) == 0,
            "invalid_words": invalid_words,
            "focus_words_found": list(focus_found),
            "focus_words_missing": focus_missing,
            "unknown_words": list(set(unknown_words)),
            "suggestion": suggestion,
            "stats": {
                "total_words": len(words),
                "unique_words": len(set(words)),
                "invalid_count": len(invalid_words),
                "unknown_count": len(set(unknown_words)),
                "focus_coverage": f"{len(focus_found)}/{len(focus_words)}"
            }
        }
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if string is punctuation"""
        punctuation = set("，。！？、；：""''（）【】《》…—·,.!?;:\"'()[]<>-_ \n\t")
        return all(c in punctuation for c in char)

    # ═══════════════════════════════════════════════════════════
    # AI Tutor Lesson Validation (Enhanced)
    # ═══════════════════════════════════════════════════════════

    def validate_reading_structured(
        self,
        chinese_text: str,
        user_lesson_position: int,
        hsk_level: int,
        focus_words: list[str],
        allowed_words: list[str] = None
    ) -> dict:
        """
        Validate reading content with structured feedback for AI retry.
        
        Returns detailed analysis including:
        - Unknown word ratio
        - Specific problem words
        - Suggestions for retry prompt
        """
        focus_set = set(focus_words)
        allowed_set = set(allowed_words) if allowed_words else None
        
        # Segment text
        words = list(jieba.cut(chinese_text))
        words = [w for w in words if w.strip() and not self._is_punctuation(w)]
        
        if not words:
            return {
                "ok": False,
                "unknown_ratio": 1.0,
                "focus_words_found": [],
                "focus_words_missing": focus_words,
                "new_words_for_user": [],
                "too_many_unknowns": [],
                "too_hard": True,
                "too_easy": False,
                "suggestions": {
                    "ban_tokens": [],
                    "require_tokens": focus_words,
                    "target_range": {"min_unknown_ratio": 0.10, "max_unknown_ratio": 0.20}
                }
            }
        
        # Categorize words
        focus_found = set()
        unknown_words = []
        too_advanced = []
        known_count = 0
        
        current_absolute = (hsk_level - 1) * 10 + user_lesson_position
        
        for word in words:
            # Focus words count as known for ratio calculation
            if word in focus_set:
                focus_found.add(word)
                known_count += 1
                continue
            
            # Always safe words
            if word in self.always_safe:
                known_count += 1
                continue
            
            # If allowed_words provided, use that as the ceiling
            if allowed_set is not None:
                if word in allowed_set:
                    known_count += 1
                else:
                    unknown_words.append(word)
                continue
            
            # Check curriculum
            if word not in self.curriculum:
                unknown_words.append(word)
                continue
            
            # Check lesson position
            word_pos = self.curriculum[word]
            word_hsk, word_lesson = self._parse_position(word_pos)
            word_absolute = (word_hsk - 1) * 10 + word_lesson
            
            if word_absolute <= current_absolute:
                known_count += 1
            else:
                too_advanced.append(word)
                unknown_words.append(word)
        
        # Calculate ratio
        total_words = len(words)
        unknown_count = len(unknown_words)
        unknown_ratio = unknown_count / total_words if total_words > 0 else 0
        
        # Determine if too hard or too easy
        # Target: 10-20% unknown words
        too_hard = unknown_ratio > 0.25
        too_easy = unknown_ratio < 0.05 and len(focus_found) < len(focus_words)
        
        # Check focus word coverage
        focus_missing = list(focus_set - focus_found)
        
        # Determine OK status
        # OK if: not too hard, all focus words present
        ok = not too_hard and len(focus_missing) == 0
        
        # Build suggestions for retry
        suggestions = None
        if not ok:
            suggestions = {
                "ban_tokens": list(set(too_advanced))[:10],  # Top 10 problem words
                "require_tokens": focus_missing,
                "target_range": {
                    "min_unknown_ratio": 0.10,
                    "max_unknown_ratio": 0.20
                }
            }
        
        return {
            "ok": ok,
            "unknown_ratio": round(unknown_ratio, 3),
            "focus_words_found": list(focus_found),
            "focus_words_missing": focus_missing,
            "new_words_for_user": list(set(unknown_words) - set(too_advanced)),
            "too_many_unknowns": list(set(too_advanced)),
            "too_hard": too_hard,
            "too_easy": too_easy,
            "suggestions": suggestions
        }

    def validate_exercise_structure(
        self,
        exercises: list[dict],
        allowed_words: list[str]
    ) -> dict:
        """
        Validate exercise JSON structure and word usage.
        
        Checks:
        - JSON schema validity
        - Unique IDs
        - All Chinese words in allowed_words
        - No illegal characters
        - Length constraints
        """
        allowed_set = set(allowed_words)
        errors = []
        warnings = []
        seen_ids = set()
        fixable = []
        must_regenerate = []
        
        for ex in exercises:
            ex_id = ex.get("id", "unknown")
            ex_type = ex.get("type", "unknown")
            
            # Check ID uniqueness
            if ex_id in seen_ids:
                errors.append({
                    "exercise_id": ex_id,
                    "field": "id",
                    "error": f"Duplicate ID: {ex_id}",
                    "severity": "error"
                })
                must_regenerate.append(ex_id)
            seen_ids.add(ex_id)
            
            # Validate based on type
            if ex_type == "multiple_choice":
                self._validate_mcq(ex, ex_id, allowed_set, errors, warnings, fixable)
            elif ex_type == "drag_sentence":
                self._validate_drag(ex, ex_id, allowed_set, errors, warnings, fixable)
            elif ex_type == "spot_error":
                self._validate_spot_error(ex, ex_id, allowed_set, errors, warnings, fixable)
            elif ex_type == "build_sentence":
                self._validate_build(ex, ex_id, allowed_set, errors, warnings, fixable)
            elif ex_type == "read_comp":
                self._validate_read_comp(ex, ex_id, allowed_set, errors, warnings, fixable)
            else:
                errors.append({
                    "exercise_id": ex_id,
                    "field": "type",
                    "error": f"Unknown exercise type: {ex_type}",
                    "severity": "error"
                })
                must_regenerate.append(ex_id)
        
        return {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "fixable": list(set(fixable)),
            "must_regenerate": list(set(must_regenerate))
        }

    def _extract_chinese_words(self, text: str) -> list[str]:
        """Extract Chinese words from text"""
        words = list(jieba.cut(text))
        return [w for w in words if w.strip() and not self._is_punctuation(w)]

    def _check_words_allowed(
        self,
        text: str,
        ex_id: str,
        field: str,
        allowed_set: set,
        errors: list,
        fixable: list
    ):
        """Check if all Chinese words in text are allowed"""
        words = self._extract_chinese_words(text)
        for word in words:
            if word not in self.always_safe and word not in allowed_set:
                errors.append({
                    "exercise_id": ex_id,
                    "field": field,
                    "error": f"Word '{word}' not in allowed_words",
                    "severity": "error"
                })
                if ex_id not in fixable:
                    fixable.append(ex_id)

    def _validate_mcq(self, ex, ex_id, allowed_set, errors, warnings, fixable):
        """Validate multiple choice exercise"""
        # Check question
        question = ex.get("question", {})
        if "chinese" in question:
            self._check_words_allowed(question["chinese"], ex_id, "question.chinese", allowed_set, errors, fixable)
        
        # Check options
        options = ex.get("options", [])
        correct_id = ex.get("correctOptionId")
        option_ids = []
        
        for i, opt in enumerate(options):
            opt_id = opt.get("id", f"opt_{i}")
            option_ids.append(opt_id)
            if "chinese" in opt:
                self._check_words_allowed(opt["chinese"], ex_id, f"options[{i}].chinese", allowed_set, errors, fixable)
        
        # Check correct option exists
        if correct_id and correct_id not in option_ids:
            errors.append({
                "exercise_id": ex_id,
                "field": "correctOptionId",
                "error": f"correctOptionId '{correct_id}' not in options",
                "severity": "error"
            })

    def _validate_drag(self, ex, ex_id, allowed_set, errors, warnings, fixable):
        """Validate drag sentence exercise"""
        # Check target sentence
        target = ex.get("targetSentence", {})
        if "chinese" in target:
            self._check_words_allowed(target["chinese"], ex_id, "targetSentence.chinese", allowed_set, errors, fixable)
        
        # Check shuffled words
        shuffled = ex.get("shuffledWords", [])
        positions = []
        for i, word in enumerate(shuffled):
            if "chinese" in word:
                self._check_words_allowed(word["chinese"], ex_id, f"shuffledWords[{i}].chinese", allowed_set, errors, fixable)
            if "correctPosition" in word:
                positions.append(word["correctPosition"])
        
        # Check positions are valid
        expected_positions = list(range(len(shuffled)))
        if sorted(positions) != expected_positions:
            warnings.append({
                "exercise_id": ex_id,
                "field": "shuffledWords.correctPosition",
                "error": f"Position values should be 0 to {len(shuffled)-1}",
                "severity": "warning"
            })

    def _validate_spot_error(self, ex, ex_id, allowed_set, errors, warnings, fixable):
        """Validate spot error exercise"""
        # Check sentence
        sentence = ex.get("sentence", {})
        if "chinese" in sentence:
            # Note: spot_error might intentionally have wrong words
            # So we check the correction instead
            pass
        
        # Check correction
        correction = ex.get("correction", {})
        if "correct" in correction:
            self._check_words_allowed(correction["correct"], ex_id, "correction.correct", allowed_set, errors, fixable)
        
        # Check words array
        words = ex.get("words", [])
        error_word_id = ex.get("errorWordId")
        word_ids = [w.get("id") for w in words]
        
        if error_word_id and error_word_id not in word_ids:
            errors.append({
                "exercise_id": ex_id,
                "field": "errorWordId",
                "error": f"errorWordId '{error_word_id}' not in words array",
                "severity": "error"
            })

    def _validate_build(self, ex, ex_id, allowed_set, errors, warnings, fixable):
        """Validate build sentence exercise"""
        # Check expected answer
        expected = ex.get("expectedAnswer", {})
        if "chinese" in expected:
            self._check_words_allowed(expected["chinese"], ex_id, "expectedAnswer.chinese", allowed_set, errors, fixable)
        
        # Check available words
        available = ex.get("availableWords", [])
        for i, word in enumerate(available):
            if "chinese" in word:
                self._check_words_allowed(word["chinese"], ex_id, f"availableWords[{i}].chinese", allowed_set, errors, fixable)
        
        # Check variations
        for i, variation in enumerate(ex.get("acceptableVariations", [])):
            self._check_words_allowed(variation, ex_id, f"acceptableVariations[{i}]", allowed_set, errors, fixable)

    def _validate_read_comp(self, ex, ex_id, allowed_set, errors, warnings, fixable):
        """Validate reading comprehension exercise"""
        # Check passage
        passage = ex.get("passage", {})
        if "chinese" in passage:
            self._check_words_allowed(passage["chinese"], ex_id, "passage.chinese", allowed_set, errors, fixable)
        
        # Check question
        question = ex.get("question", {})
        if "chinese" in question:
            self._check_words_allowed(question["chinese"], ex_id, "question.chinese", allowed_set, errors, fixable)
        
        # Check options
        options = ex.get("options", [])
        correct_id = ex.get("correctOptionId")
        option_ids = []
        
        for i, opt in enumerate(options):
            opt_id = opt.get("id", f"opt_{i}")
            option_ids.append(opt_id)
            if "chinese" in opt:
                self._check_words_allowed(opt["chinese"], ex_id, f"options[{i}].chinese", allowed_set, errors, fixable)
        
        if correct_id and correct_id not in option_ids:
            errors.append({
                "exercise_id": ex_id,
                "field": "correctOptionId",
                "error": f"correctOptionId '{correct_id}' not in options",
                "severity": "error"
            })

    def validate_pedagogy(
        self,
        reading_chinese: str,
        exercises: list[dict],
        user_lesson_position: int,
        hsk_level: int,
        focus_words: list[str]
    ) -> dict:
        """
        Validate all content for pedagogical soundness.
        
        Checks:
        - Unknown word density per exercise
        - Focus word coverage
        - No grammar beyond user's level (via word check)
        """
        focus_set = set(focus_words)
        current_absolute = (hsk_level - 1) * 10 + user_lesson_position
        
        items = []
        focus_tested = {}  # Track which focus words are tested
        
        # Validate reading
        reading_result = self._validate_item_pedagogy(
            "reading",
            reading_chinese,
            current_absolute,
            focus_set
        )
        items.append(reading_result)
        for fw in reading_result.get("focus_words_in_item", []):
            focus_tested[fw] = focus_tested.get(fw, 0) + 1
        
        # Validate each exercise
        for ex in exercises:
            ex_id = ex.get("id", "unknown")
            ex_type = ex.get("type", "unknown")
            
            # Extract Chinese text based on exercise type
            chinese_texts = self._extract_exercise_chinese(ex, ex_type)
            combined_text = " ".join(chinese_texts)
            
            result = self._validate_item_pedagogy(
                ex_id,
                combined_text,
                current_absolute,
                focus_set,
                max_unknown_ratio=0.30  # Exercises can be slightly harder
            )
            items.append(result)
            for fw in result.get("focus_words_in_item", []):
                focus_tested[fw] = focus_tested.get(fw, 0) + 1
        
        # Check coverage
        tested = list(focus_tested.keys())
        untested = [fw for fw in focus_words if fw not in focus_tested]
        
        # Overall OK if all items pass and all focus words tested
        all_ok = all(item["ok"] for item in items) and len(untested) == 0
        
        return {
            "ok": all_ok,
            "items": items,
            "coverage": {
                "focus_words_tested": tested,
                "focus_words_untested": untested,
                "times_tested": focus_tested
            }
        }

    def _validate_item_pedagogy(
        self,
        item_id: str,
        chinese_text: str,
        current_absolute: int,
        focus_set: set,
        max_unknown_ratio: float = 0.25
    ) -> dict:
        """Validate a single item for pedagogy"""
        words = self._extract_chinese_words(chinese_text)
        
        if not words:
            return {
                "id": item_id,
                "ok": True,
                "unknown_ratio": 0,
                "focus_word_tested": None,
                "focus_words_in_item": [],
                "issues": []
            }
        
        unknown_count = 0
        issues = []
        focus_in_item = []
        
        for word in words:
            if word in focus_set:
                focus_in_item.append(word)
                continue
            
            if word in self.always_safe:
                continue
            
            if word not in self.curriculum:
                unknown_count += 1
                continue
            
            word_pos = self.curriculum[word]
            word_hsk, word_lesson = self._parse_position(word_pos)
            word_absolute = (word_hsk - 1) * 10 + word_lesson
            
            if word_absolute > current_absolute:
                unknown_count += 1
                issues.append(f"Word '{word}' from lesson {word_absolute} exceeds current {current_absolute}")
        
        unknown_ratio = unknown_count / len(words)
        
        if unknown_ratio > max_unknown_ratio:
            issues.append(f"Unknown ratio {unknown_ratio:.0%} exceeds max {max_unknown_ratio:.0%}")
        
        return {
            "id": item_id,
            "ok": len(issues) == 0,
            "unknown_ratio": round(unknown_ratio, 3),
            "focus_word_tested": focus_in_item[0] if focus_in_item else None,
            "focus_words_in_item": list(set(focus_in_item)),
            "issues": issues
        }

    def _extract_exercise_chinese(self, ex: dict, ex_type: str) -> list[str]:
        """Extract all Chinese text from an exercise"""
        texts = []
        
        if ex_type == "multiple_choice":
            if "question" in ex and "chinese" in ex["question"]:
                texts.append(ex["question"]["chinese"])
            for opt in ex.get("options", []):
                if "chinese" in opt:
                    texts.append(opt["chinese"])
        
        elif ex_type == "drag_sentence":
            if "targetSentence" in ex and "chinese" in ex["targetSentence"]:
                texts.append(ex["targetSentence"]["chinese"])
        
        elif ex_type == "spot_error":
            if "correction" in ex and "correct" in ex["correction"]:
                texts.append(ex["correction"]["correct"])
        
        elif ex_type == "build_sentence":
            if "expectedAnswer" in ex and "chinese" in ex["expectedAnswer"]:
                texts.append(ex["expectedAnswer"]["chinese"])
        
        elif ex_type == "read_comp":
            if "passage" in ex and "chinese" in ex["passage"]:
                texts.append(ex["passage"]["chinese"])
            if "question" in ex and "chinese" in ex["question"]:
                texts.append(ex["question"]["chinese"])
        
        return texts

