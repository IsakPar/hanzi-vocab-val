"""
Pydantic models for the vocab-validator service.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Set
from enum import Enum


# ═══════════════════════════════════════════════════════════
# Tier Configuration
# ═══════════════════════════════════════════════════════════

class TierName(str, Enum):
    COMFORT = "comfort"
    CHALLENGE = "challenge"
    STRETCH = "stretch"


TIER_CONFIG = {
    TierName.COMFORT: {
        "min": 0.95,
        "max": 1.00,
        "label": "Comfort Zone",
        "description": "You know 95%+ of words - perfect for building confidence",
        "emoji": "🟢",
    },
    TierName.CHALLENGE: {
        "min": 0.85,
        "max": 0.95,
        "label": "Sweet Spot",
        "description": "Optimal learning zone - challenging but achievable",
        "emoji": "🟡",
    },
    TierName.STRETCH: {
        "min": 0.75,
        "max": 0.85,
        "label": "Stretch Goal",
        "description": "Ambitious read - use dictionary support",
        "emoji": "🔴",
    },
}


# ═══════════════════════════════════════════════════════════
# Curriculum Models
# ═══════════════════════════════════════════════════════════

class VocabWord(BaseModel):
    id: str
    hanzi: str
    pinyin: str
    hskLevel: int


class Lesson(BaseModel):
    id: str
    hskLevel: int
    lessonNumber: int
    title: str
    targetVocabulary: List[str]  # vocab IDs


class Token(BaseModel):
    wordId: Optional[str]
    hanzi: str


class Story(BaseModel):
    id: str
    title: str
    hskLevel: int
    difficulty: str
    tokens: List[Token]  # Pre-tokenized by backend
    totalTokens: int
    sentenceCount: int


class Audiobook(BaseModel):
    id: str
    title: str
    hskLevel: int
    description: Optional[str]
    tokens: List[Token]  # Pre-tokenized by backend
    totalTokens: int


class FullCurriculum(BaseModel):
    version: str
    exportedAt: str
    vocabulary: List[VocabWord]
    lessons: List[Lesson]
    lessonOrder: List[str]
    lessonWordMap: Dict[str, List[str]]
    stories: List[Story]
    audiobooks: List[Audiobook]


# ═══════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    lesson_id: str  # e.g., "lesson-uuid" or "hsk1-l5"
    content_type: str = "all"  # "story", "audiobook", "all"
    items_per_tier: int = 3


class UnknownWord(BaseModel):
    wordId: str
    hanzi: str


class ContentItem(BaseModel):
    type: str  # "story" or "audiobook"
    id: str
    title: str
    hskLevel: int
    comprehension: float  # 0.0 - 1.0
    totalTokens: int
    unknownWords: List[UnknownWord]
    unknownCount: int


class TierResult(BaseModel):
    label: str
    description: str
    emoji: str
    range: List[float]  # [min, max]
    items: List[ContentItem]


class RecommendResponse(BaseModel):
    lessonId: str
    knownWordCount: int
    contentType: str
    tiers: Dict[str, TierResult]
    excludedCount: int  # Items below 75%
    generatedAt: str

