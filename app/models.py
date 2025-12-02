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


# ═══════════════════════════════════════════════════════════
# AI Tutor Lesson Validation Models
# ═══════════════════════════════════════════════════════════

class ReadingSentence(BaseModel):
    id: str
    chinese: str
    pinyin: str
    english: str


class ReadingContent(BaseModel):
    id: str
    chinese: str
    pinyin: str
    english: str
    sentences: List[ReadingSentence] = []


class ReadingValidationSuggestions(BaseModel):
    ban_tokens: List[str] = []
    require_tokens: List[str] = []
    target_range: Dict[str, float] = {"min_unknown_ratio": 0.10, "max_unknown_ratio": 0.20}


class ValidateReadingRequest(BaseModel):
    """Request for reading validation with structured feedback"""
    reading: ReadingContent
    user_lesson_position: int  # Absolute lesson number (1-based)
    hsk_level: int
    focus_words: List[str]
    allowed_words: List[str] = []  # Optional pre-computed allowed words


class ValidateReadingResponse(BaseModel):
    """Response with structured feedback for AI retry"""
    ok: bool
    unknown_ratio: float
    focus_words_found: List[str]
    focus_words_missing: List[str]
    new_words_for_user: List[str]  # Words user hasn't seen but are valid
    too_many_unknowns: List[str] = []  # Specific problem words
    too_hard: bool = False
    too_easy: bool = False
    suggestions: Optional[ReadingValidationSuggestions] = None


# ═══════════════════════════════════════════════════════════
# Structural Validation Models
# ═══════════════════════════════════════════════════════════

class ExerciseOption(BaseModel):
    id: str
    chinese: str
    pinyin: str
    english: Optional[str] = None


class MultipleChoiceExercise(BaseModel):
    id: str
    type: str = "multiple_choice"
    question: Dict[str, str]
    options: List[ExerciseOption]
    correctOptionId: str


class DragSentenceWord(BaseModel):
    id: str
    chinese: str
    pinyin: str
    correctPosition: int


class DragSentenceExercise(BaseModel):
    id: str
    type: str = "drag_sentence"
    targetSentence: Dict[str, str]
    shuffledWords: List[DragSentenceWord]


class SpotErrorWord(BaseModel):
    id: str
    chinese: str
    isError: bool


class SpotErrorExercise(BaseModel):
    id: str
    type: str = "spot_error"
    sentence: Dict[str, str]
    errorWordId: str
    correction: Dict[str, str]
    words: List[SpotErrorWord]


class BuildSentenceWord(BaseModel):
    id: str
    chinese: str
    pinyin: str


class BuildSentenceExercise(BaseModel):
    id: str
    type: str = "build_sentence"
    prompt: Dict[str, str]
    expectedAnswer: Dict[str, str]
    acceptableVariations: List[str] = []
    availableWords: List[BuildSentenceWord]


class ReadCompExercise(BaseModel):
    id: str
    type: str = "read_comp"
    passage: Dict[str, str]
    question: Dict[str, str]
    options: List[ExerciseOption]
    correctOptionId: str


class StructuralError(BaseModel):
    exercise_id: str
    field: str
    error: str
    severity: str  # "error" or "warning"


class ValidateStructureRequest(BaseModel):
    """Request for structural validation"""
    exercises: List[Dict]  # Raw exercise data
    allowed_words: List[str]


class ValidateStructureResponse(BaseModel):
    """Response for structural validation"""
    ok: bool
    errors: List[StructuralError] = []
    warnings: List[StructuralError] = []
    fixable: List[str] = []  # Exercise IDs that can be fixed
    must_regenerate: List[str] = []  # Exercise IDs that need regeneration


# ═══════════════════════════════════════════════════════════
# Pedagogical Validation Models
# ═══════════════════════════════════════════════════════════

class PedagogyItemResult(BaseModel):
    id: str
    ok: bool
    unknown_ratio: Optional[float] = None
    focus_word_tested: Optional[str] = None
    issues: List[str] = []


class FocusWordCoverage(BaseModel):
    focus_words_tested: List[str]
    focus_words_untested: List[str]
    times_tested: Dict[str, int]


class ValidatePedagogyRequest(BaseModel):
    """Request for pedagogical validation"""
    reading: ReadingContent
    exercises: List[Dict]
    user_lesson_position: int
    hsk_level: int
    focus_words: List[str]


class ValidatePedagogyResponse(BaseModel):
    """Response for pedagogical validation"""
    ok: bool
    items: List[PedagogyItemResult]
    coverage: FocusWordCoverage

