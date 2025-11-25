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
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if string is punctuation"""
        punctuation = set("，。！？、；：""''（）【】《》…—·,.!?;:\"'()[]<>-_ \n\t")
        return all(c in punctuation for c in char)

