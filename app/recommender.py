"""
Content Recommendation Engine

Recommends stories and audiobooks based on user's vocabulary level.
Uses token-level comprehension (total occurrences, not unique types).
Returns tiered recommendations: Comfort (95%+), Challenge (85-94%), Stretch (75-84%).
"""

import json
import os
import jieba
import logging
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

from .models import (
    TierName, TIER_CONFIG,
    ContentItem, UnknownWord, TierResult, RecommendResponse,
    VocabWord, Story, Audiobook
)

logger = logging.getLogger(__name__)


class ContentRecommender:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.loaded = False
        
        # Curriculum data
        self.vocab_by_id: Dict[str, VocabWord] = {}
        self.vocab_by_hanzi: Dict[str, str] = {}  # hanzi -> vocab_id
        self.lesson_order: List[str] = []
        self.lesson_word_map: Dict[str, List[str]] = {}  # lesson_id -> vocab_ids
        
        # Cumulative word sets per lesson (computed once)
        self.cumulative_words: Dict[str, Set[str]] = {}  # lesson_id -> set of vocab_ids
        
        # Content with pre-tokenized words
        self.stories: List[Dict] = []  # Story with tokenized words
        self.audiobooks: List[Dict] = []  # Audiobook with tokenized words
        
        self.version: str = ""
    
    def load(self):
        """Load full curriculum from cache and compute cumulative word sets."""
        content_path = os.path.join(self.data_dir, "content.json")
        
        if not os.path.exists(content_path):
            logger.warning("Content cache not found - recommender unavailable")
            return
        
        with open(content_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Load vocabulary
        for v in data.get("vocabulary", []):
            vocab = VocabWord(**v)
            self.vocab_by_id[vocab.id] = vocab
            self.vocab_by_hanzi[vocab.hanzi] = vocab.id
            # Add to jieba for proper segmentation
            jieba.add_word(vocab.hanzi)
        
        # Load lesson structure
        self.lesson_order = data.get("lessonOrder", [])
        self.lesson_word_map = data.get("lessonWordMap", {})
        
        # Compute cumulative word sets
        self._build_cumulative_words()
        
        # Load and tokenize stories
        self.stories = []
        for story in data.get("stories", []):
            tokenized = self._tokenize_content(story["fullText"])
            self.stories.append({
                "id": story["id"],
                "title": story["title"],
                "hskLevel": story["hskLevel"],
                "difficulty": story.get("difficulty", "medium"),
                "tokens": tokenized,
                "totalTokens": len(tokenized),
                "type": "story"
            })
        
        # Load and tokenize audiobooks
        self.audiobooks = []
        for ab in data.get("audiobooks", []):
            if ab.get("fullText"):  # Only if has transcript
                tokenized = self._tokenize_content(ab["fullText"])
                self.audiobooks.append({
                    "id": ab["id"],
                    "title": ab["title"],
                    "hskLevel": ab["hskLevel"],
                    "tokens": tokenized,
                    "totalTokens": len(tokenized),
                    "type": "audiobook"
                })
        
        self.version = data.get("version", "")
        self.loaded = True
        
        logger.info(
            f"Recommender loaded: {len(self.vocab_by_id)} words, "
            f"{len(self.lesson_order)} lessons, "
            f"{len(self.stories)} stories, {len(self.audiobooks)} audiobooks"
        )
    
    def reload(self):
        """Reload after sync."""
        self.load()
    
    def _build_cumulative_words(self):
        """Build cumulative word sets for each lesson position."""
        self.cumulative_words = {}
        known_so_far: Set[str] = set()
        
        for lesson_id in self.lesson_order:
            new_words = set(self.lesson_word_map.get(lesson_id, []))
            known_so_far = known_so_far | new_words
            self.cumulative_words[lesson_id] = known_so_far.copy()
        
        logger.info(f"Built cumulative word sets for {len(self.cumulative_words)} lessons")
    
    def _tokenize_content(self, text: str) -> List[Dict]:
        """
        Tokenize Chinese text and map to vocabulary IDs.
        
        Returns list of tokens: [{"wordId": "vocab-123" or None, "hanzi": "你好"}, ...]
        """
        if not text:
            return []
        
        segments = list(jieba.cut(text))
        tokens = []
        
        for segment in segments:
            # Skip punctuation and whitespace
            if self._is_punctuation(segment):
                continue
            
            word_id = self.vocab_by_hanzi.get(segment)
            tokens.append({
                "wordId": word_id,
                "hanzi": segment
            })
        
        return tokens
    
    def _is_punctuation(self, char: str) -> bool:
        """Check if string is punctuation or whitespace."""
        punctuation = set("，。！？、；：""''（）【】《》…—·,.!?;:\"'()[]<>-_ \n\t\r")
        return all(c in punctuation for c in char)
    
    def _calculate_comprehension(
        self, 
        tokens: List[Dict], 
        known_word_ids: Set[str]
    ) -> Tuple[float, List[Dict], int]:
        """
        Calculate token-level comprehension.
        
        Returns:
        - comprehension: float (0.0 - 1.0)
        - unknown_words: list of unique unknown words (as dicts)
        - unknown_count: total unique unknown words
        """
        # Only count curriculum tokens (those with wordId)
        curriculum_tokens = [t for t in tokens if t["wordId"] is not None]
        
        if not curriculum_tokens:
            return 1.0, [], 0
        
        known_count = sum(1 for t in curriculum_tokens if t["wordId"] in known_word_ids)
        comprehension = known_count / len(curriculum_tokens)
        
        # Get unique unknown words
        unknown_ids = set()
        unknown_words = []
        for t in curriculum_tokens:
            if t["wordId"] not in known_word_ids and t["wordId"] not in unknown_ids:
                unknown_ids.add(t["wordId"])
                unknown_words.append({
                    "wordId": t["wordId"],
                    "hanzi": t["hanzi"]
                })
        
        return comprehension, unknown_words, len(unknown_words)
    
    def get_known_words_for_lesson(self, lesson_id: str) -> Set[str]:
        """Get all word IDs known at a specific lesson position."""
        if lesson_id in self.cumulative_words:
            return self.cumulative_words[lesson_id]
        
        # If lesson_id not found, try to find closest match
        # (user might be at a lesson we don't have)
        logger.warning(f"Lesson ID not found: {lesson_id}")
        return set()
    
    def recommend(
        self,
        lesson_id: str,
        content_type: str = "all",
        items_per_tier: int = 3
    ) -> RecommendResponse:
        """
        Get tiered content recommendations based on user's lesson position.
        
        Args:
            lesson_id: User's current lesson (e.g., "lesson-uuid")
            content_type: "story", "audiobook", or "all"
            items_per_tier: Max items per tier (default 3)
        
        Returns:
            RecommendResponse with tiered recommendations
        """
        known_word_ids = self.get_known_words_for_lesson(lesson_id)
        
        # Gather all content
        all_content = []
        
        if content_type in ("story", "all"):
            for story in self.stories:
                comp, unknown, unknown_count = self._calculate_comprehension(
                    story["tokens"], known_word_ids
                )
                all_content.append({
                    "type": "story",
                    "id": story["id"],
                    "title": story["title"],
                    "hskLevel": story["hskLevel"],
                    "comprehension": comp,
                    "totalTokens": story["totalTokens"],
                    "unknownWords": unknown[:5],  # Preview limit
                    "unknownCount": unknown_count,
                })
        
        if content_type in ("audiobook", "all"):
            for ab in self.audiobooks:
                comp, unknown, unknown_count = self._calculate_comprehension(
                    ab["tokens"], known_word_ids
                )
                all_content.append({
                    "type": "audiobook",
                    "id": ab["id"],
                    "title": ab["title"],
                    "hskLevel": ab["hskLevel"],
                    "comprehension": comp,
                    "totalTokens": ab["totalTokens"],
                    "unknownWords": unknown[:5],
                    "unknownCount": unknown_count,
                })
        
        # Sort by comprehension descending
        all_content.sort(key=lambda x: -x["comprehension"])
        
        # Build tiered response
        tiers = {}
        excluded_count = 0
        
        for tier_name in TierName:
            config = TIER_CONFIG[tier_name]
            tier_items = [
                ContentItem(
                    type=c["type"],
                    id=c["id"],
                    title=c["title"],
                    hskLevel=c["hskLevel"],
                    comprehension=round(c["comprehension"], 3),
                    totalTokens=c["totalTokens"],
                    unknownWords=[UnknownWord(**w) for w in c["unknownWords"]],
                    unknownCount=c["unknownCount"],
                )
                for c in all_content
                if config["min"] <= c["comprehension"] < config["max"]
            ][:items_per_tier]
            
            tiers[tier_name.value] = TierResult(
                label=config["label"],
                description=config["description"],
                emoji=config["emoji"],
                range=[config["min"], config["max"]],
                items=tier_items,
            )
        
        # Count excluded (below 75%)
        excluded_count = sum(1 for c in all_content if c["comprehension"] < 0.75)
        
        return RecommendResponse(
            lessonId=lesson_id,
            knownWordCount=len(known_word_ids),
            contentType=content_type,
            tiers=tiers,
            excludedCount=excluded_count,
            generatedAt=datetime.utcnow().isoformat() + "Z",
        )
    
    def get_info(self) -> dict:
        """Get recommender status info."""
        return {
            "loaded": self.loaded,
            "vocab_count": len(self.vocab_by_id),
            "lesson_count": len(self.lesson_order),
            "story_count": len(self.stories),
            "audiobook_count": len(self.audiobooks),
            "version": self.version,
        }

