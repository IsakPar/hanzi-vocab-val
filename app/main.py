"""
Vocabulary Validator Service
FastAPI service for validating AI-generated Chinese content against curriculum.

Security:
- CORS locked to known origins only
- /sync endpoint protected by API key
- Auto-sync on startup if no cache
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv

from .validator import VocabValidator
from .sync import CurriculumSync
from .recommender import ContentRecommender
from .models import (
    RecommendRequest, RecommendResponse,
    ValidateReadingRequest, ValidateReadingResponse,
    ValidateStructureRequest, ValidateStructureResponse,
    ValidatePedagogyRequest, ValidatePedagogyResponse
)

load_dotenv()

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

BACKEND_URL = os.getenv("BACKEND_URL", "https://hanzimaster-backend-v2.isak-parild.workers.dev")
DATA_DIR = os.getenv("DATA_DIR", "./data")
API_KEY = os.getenv("VALIDATOR_API_KEY", "")  # Required for /sync
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Allowed origins - backend + portal + localhost
ALLOWED_ORIGINS = [
    BACKEND_URL,
    "https://hanzimaster-backend-v2.isak-parild.workers.dev",
    # Portal URLs (including all deployment variants)
    "https://hanzimaster-portal.pages.dev",
    "https://hanzimaster-portal-v2.pages.dev",
    "https://hanzimaster-studio.pages.dev",  # Production studio URL
    # Cloudflare preview deployments (wildcard would be better but FastAPI doesn't support it well)
    # Localhost for development
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8787",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# Initialize Services
# ═══════════════════════════════════════════════════════════

sync = CurriculumSync(backend_url=BACKEND_URL, data_dir=DATA_DIR)
validator = VocabValidator(data_dir=DATA_DIR)
recommender = ContentRecommender(data_dir=DATA_DIR)


# ═══════════════════════════════════════════════════════════
# Lifespan (startup/shutdown)
# ═══════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info(f"Starting vocab-validator (env: {ENVIRONMENT})")
    logger.info(f"Backend URL: {BACKEND_URL}")
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Load validator
    try:
        validator.load()
        logger.info(f"✓ Curriculum loaded: {validator.get_curriculum_info()['word_count']} words")
    except FileNotFoundError:
        logger.warning("No curriculum cache found - attempting auto-sync...")
        try:
            result = await sync.sync()
            if result["success"]:
                validator.load()
                logger.info(f"✓ Auto-sync successful: {result['word_count']} words")
            else:
                logger.error("Auto-sync failed - service will start without curriculum")
        except Exception as e:
            logger.error(f"Auto-sync error: {e}")
            logger.warning("Service starting without curriculum - POST /sync to initialize")
    except Exception as e:
        logger.error(f"Failed to load curriculum: {e}")
    
    # Load recommender
    try:
        recommender.load()
        info = recommender.get_info()
        logger.info(
            f"✓ Recommender loaded: {info['story_count']} stories, "
            f"{info['audiobook_count']} audiobooks"
        )
    except FileNotFoundError:
        logger.warning("No content cache found - attempting auto-sync...")
        try:
            result = await sync.sync_full()
            if result["success"] and result["changed"]:
                recommender.load()
                logger.info(f"✓ Content auto-sync successful")
            else:
                logger.info("No content available - recommender will be unavailable")
        except Exception as e:
            logger.error(f"Content auto-sync error: {e}")
    except Exception as e:
        logger.error(f"Failed to load recommender: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down vocab-validator")


app = FastAPI(
    title="Vocab Validator",
    description="Validates Chinese text against curriculum vocabulary",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - locked down
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# Security Dependencies
# ═══════════════════════════════════════════════════════════

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        # No API key configured - allow in development
        if ENVIRONMENT == "development":
            return True
        raise HTTPException(
            status_code=500, 
            detail="VALIDATOR_API_KEY not configured"
        )
    
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, 
            detail="Invalid or missing API key"
        )
    return True


# ═══════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════

class UserPosition(BaseModel):
    hsk: int
    lesson: int


class ValidateRequest(BaseModel):
    text: str
    user_position: UserPosition
    target_words: list[str] = []


class ValidateResponse(BaseModel):
    valid: bool
    words_found: list[str]
    safe_words: list[str]
    target_words: list[str]
    forbidden_words: list[str]
    unknown_words: list[str]
    stats: dict


# ═══════════════════════════════════════════════════════════
# i+1 Lesson Validation Models
# ═══════════════════════════════════════════════════════════

class ValidateLessonRequest(BaseModel):
    """Request for i+1 lesson validation"""
    text: str  # The Chinese lesson text
    lesson_number: int  # Current lesson number (1-based)
    focus_words: list[str]  # New words being taught (i+1)
    hsk_level: int = 1  # HSK level (default 1)


class InvalidWord(BaseModel):
    """Details about a word that violates i+1"""
    word: str
    lesson_id: int  # The lesson this word belongs to
    reason: str  # Why it's invalid


class ValidateLessonResponse(BaseModel):
    """Response for i+1 lesson validation"""
    valid: bool
    invalid_words: list[InvalidWord] = []
    focus_words_found: list[str] = []  # Which focus words appeared
    focus_words_missing: list[str] = []  # Focus words not in text
    unknown_words: list[str] = []  # Words not in curriculum
    suggestion: Optional[str] = None  # AI feedback for retry
    stats: dict = {}


class SyncResponse(BaseModel):
    success: bool
    version: str
    word_count: int
    lesson_count: int
    changed: bool


class HealthResponse(BaseModel):
    status: str
    curriculum_loaded: bool
    word_count: int
    version: str
    environment: str


# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - public"""
    curriculum = validator.get_curriculum_info()
    return HealthResponse(
        status="healthy" if curriculum["loaded"] else "degraded",
        curriculum_loaded=curriculum["loaded"],
        word_count=curriculum["word_count"],
        version=curriculum["version"],
        environment=ENVIRONMENT
    )


@app.post("/validate", response_model=ValidateResponse)
async def validate_text(request: ValidateRequest):
    """
    Validate Chinese text against user's curriculum position.
    
    Logic:
    - Words from lessons BEFORE user_position = SAFE
    - Words at user_position = TARGET (learning now)
    - Words AFTER user_position = FORBIDDEN
    - Words not in curriculum = UNKNOWN
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503, 
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        result = validator.validate(
            text=request.text,
            user_hsk=request.user_position.hsk,
            user_lesson=request.user_position.lesson,
            target_words=request.target_words
        )
        return ValidateResponse(**result)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# i+1 Lesson Validation Endpoint
# ═══════════════════════════════════════════════════════════

@app.post("/validate-lesson", response_model=ValidateLessonResponse)
async def validate_lesson(request: ValidateLessonRequest):
    """
    Validate lesson text for strict i+1 compliance.
    
    Rules:
    - ALL words must have lesson_id ≤ current lesson number
    - EXCEPT focus_words (the new i+1 vocabulary)
    - Unknown words (not in curriculum) are flagged
    
    This ensures students only see known vocabulary plus
    the specific new words being taught.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        result = validator.validate_lesson(
            text=request.text,
            lesson_number=request.lesson_number,
            focus_words=request.focus_words,
            hsk_level=request.hsk_level
        )
        return ValidateLessonResponse(**result)
    except Exception as e:
        logger.error(f"Lesson validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# AI Tutor Lesson Validation Endpoints (Enhanced)
# ═══════════════════════════════════════════════════════════

@app.post("/validate/reading", response_model=ValidateReadingResponse)
async def validate_reading_structured(request: ValidateReadingRequest):
    """
    Validate reading content with structured feedback for AI retry.
    
    Returns detailed analysis including:
    - Unknown word ratio
    - Specific problem words
    - Suggestions for retry prompt (ban_tokens, require_tokens)
    
    This is used by the AI Tutor lesson generator to validate
    reading content before generating practice exercises.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        result = validator.validate_reading_structured(
            chinese_text=request.reading.chinese,
            user_lesson_position=request.user_lesson_position,
            hsk_level=request.hsk_level,
            focus_words=request.focus_words,
            allowed_words=request.allowed_words if request.allowed_words else None
        )
        return ValidateReadingResponse(**result)
    except Exception as e:
        logger.error(f"Reading validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/structure", response_model=ValidateStructureResponse)
async def validate_structure(request: ValidateStructureRequest):
    """
    Validate exercise JSON structure and word usage.
    
    Checks:
    - JSON schema validity
    - Unique IDs
    - All Chinese words in allowed_words
    - No illegal characters
    - Length constraints
    
    Returns which exercises can be fixed vs must be regenerated.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        result = validator.validate_exercise_structure(
            exercises=request.exercises,
            allowed_words=request.allowed_words
        )
        return ValidateStructureResponse(**result)
    except Exception as e:
        logger.error(f"Structure validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/pedagogy", response_model=ValidatePedagogyResponse)
async def validate_pedagogy(request: ValidatePedagogyRequest):
    """
    Validate all content for pedagogical soundness (i+1 compliance).
    
    Checks:
    - Unknown word density per exercise (max 30%)
    - Unknown word density for reading (max 25%)
    - Focus word coverage (all focus words tested)
    - No grammar beyond user's level
    
    Returns per-item results and overall coverage analysis.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        result = validator.validate_pedagogy(
            reading_chinese=request.reading.chinese,
            exercises=request.exercises,
            user_lesson_position=request.user_lesson_position,
            hsk_level=request.hsk_level,
            focus_words=request.focus_words
        )
        return ValidatePedagogyResponse(**result)
    except Exception as e:
        logger.error(f"Pedagogy validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync", response_model=SyncResponse, dependencies=[Depends(verify_api_key)])
async def sync_curriculum():
    """
    Sync curriculum from backend.
    
    PROTECTED: Requires X-API-Key header.
    Called daily or manually to update local cache.
    """
    logger.info("Sync requested")
    try:
        result = await sync.sync()
        if result["changed"]:
            validator.reload()
            logger.info(f"Curriculum reloaded: {result['word_count']} words")
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version")
async def get_version():
    """Get current curriculum version - public"""
    curriculum = validator.get_curriculum_info()
    rec_info = recommender.get_info()
    return {
        "version": curriculum["version"],
        "word_count": curriculum["word_count"],
        "loaded": curriculum["loaded"],
        "recommender": {
            "loaded": rec_info["loaded"],
            "story_count": rec_info["story_count"],
            "audiobook_count": rec_info["audiobook_count"],
        }
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_content(request: RecommendRequest):
    """
    Get tiered content recommendations based on user's lesson position.
    
    Returns content in 3 tiers:
    - 🟢 Comfort Zone (95%+): Build confidence
    - 🟡 Sweet Spot (85-94%): Optimal learning
    - 🔴 Stretch Goal (75-84%): Ambitious with support
    
    Content below 75% comprehension is excluded.
    """
    if not recommender.loaded:
        raise HTTPException(
            status_code=503,
            detail="Recommender not loaded. POST /sync to initialize."
        )
    
    try:
        result = recommender.recommend(
            lesson_id=request.lesson_id,
            content_type=request.content_type,
            items_per_tier=request.items_per_tier,
        )
        return result
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# GET VOCABULARY (for generation prompts)
# ═══════════════════════════════════════════════════════════

@app.get("/get-vocabulary")
async def get_vocabulary(max_lesson: int = 10):
    """
    Get all vocabulary words up to a given lesson number.
    Used by lesson generator to know which words are allowed.
    
    Returns words from lesson 1 to max_lesson inclusive.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    try:
        allowed_words = []
        
        for word, position in validator.curriculum.items():
            # Parse position like "hsk1-l5" to extract lesson number
            try:
                # Format: "hsk{X}-l{Y}" where X is HSK level, Y is lesson in that level
                parts = position.lower().replace("hsk", "").split("-l")
                if len(parts) == 2:
                    hsk_level = int(parts[0])
                    lesson_in_hsk = int(parts[1])
                    # Calculate absolute lesson number
                    # HSK 1 = lessons 1-10, HSK 2 = lessons 11-20, etc.
                    absolute_lesson = (hsk_level - 1) * 10 + lesson_in_hsk
                    
                    if absolute_lesson <= max_lesson:
                        allowed_words.append(word)
            except (ValueError, IndexError):
                # If position parsing fails, skip this word
                continue
        
        logger.info(f"Returning {len(allowed_words)} words for max_lesson={max_lesson}")
        
        return {
            "words": allowed_words,
            "count": len(allowed_words),
            "max_lesson": max_lesson
        }
    except Exception as e:
        logger.error(f"Get vocabulary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# SEGMENTATION (for Portal health check)
# ═══════════════════════════════════════════════════════════

class SegmentRequest(BaseModel):
    """Request to segment Chinese text into words"""
    texts: list[str]  # Multiple Chinese texts to segment


class SegmentedText(BaseModel):
    """Single segmented text result"""
    text: str
    words: list[str]


class SegmentResponse(BaseModel):
    """Response with segmented words"""
    segments: list[SegmentedText]
    all_words: list[str]  # Deduplicated list of all words
    words_filtered: list[str]  # After removing always_safe words
    always_safe_removed: list[str]  # Which words were filtered out
    curriculum_words: list[str]  # Words that exist in curriculum
    unknown_words: list[str]  # Words NOT in curriculum (candidates for adding)


@app.post("/segment", response_model=SegmentResponse)
async def segment_text(request: SegmentRequest):
    """
    Segment Chinese text into individual words using jieba.
    
    Used by Portal health check to:
    1. Break sentences like "我是中国人" into ["我", "是", "中国", "人"]
    2. Filter out always_safe words (我, 你, 是, 的, etc.)
    3. Identify which words exist in curriculum vs unknown
    
    This allows health check to flag only ACTUAL missing vocabulary,
    not entire sentences.
    """
    if not validator.loaded:
        raise HTTPException(
            status_code=503,
            detail="Curriculum not loaded. POST /sync to initialize."
        )
    
    all_words = set()
    always_safe_removed = set()
    segments = []
    
    for text in request.texts:
        # Use validator's extraction method (includes jieba + punctuation filtering)
        words = validator._extract_chinese_words(text)
        segments.append(SegmentedText(text=text, words=words))
        all_words.update(words)
    
    # Separate always_safe from others
    words_filtered = []
    for word in all_words:
        if word in validator.always_safe:
            always_safe_removed.add(word)
        else:
            words_filtered.append(word)
    
    # Check which words are in curriculum
    curriculum_words = []
    unknown_words = []
    for word in words_filtered:
        if word in validator.curriculum:
            curriculum_words.append(word)
        else:
            unknown_words.append(word)
    
    return SegmentResponse(
        segments=segments,
        all_words=list(all_words),
        words_filtered=words_filtered,
        always_safe_removed=list(always_safe_removed),
        curriculum_words=curriculum_words,
        unknown_words=unknown_words
    )


# ═══════════════════════════════════════════════════════════
# TEST SEEDING (for pipeline testing)
# ═══════════════════════════════════════════════════════════

class SeedTestCurriculumRequest(BaseModel):
    """Request to seed test curriculum"""
    words: dict  # {"你好": "hsk1-l1", "学习": "hsk1-l3", ...}


@app.post("/seed-test-curriculum")
async def seed_test_curriculum(request: SeedTestCurriculumRequest):
    """
    Seed the validator with test curriculum data.
    This is for pipeline testing only - NOT for production.
    
    Accepts a dict of {word: position} and loads it directly
    into the validator without persisting to disk.
    """
    import jieba
    
    # Load directly into validator memory
    validator.curriculum = request.words
    validator.version = "test-seed"
    validator.loaded = True
    
    # Add all words to jieba for proper segmentation
    for word in validator.curriculum.keys():
        jieba.add_word(word, freq=1000)
    
    logger.info(f"Test curriculum seeded: {len(request.words)} words")
    
    return {
        "success": True,
        "word_count": len(request.words),
        "version": "test-seed",
        "sample_words": list(request.words.items())[:5]
    }


@app.post("/sync/full", dependencies=[Depends(verify_api_key)])
async def sync_all_content():
    """
    Sync both curriculum and full content from backend.
    
    PROTECTED: Requires X-API-Key header.
    Use this to update both validator and recommender data.
    """
    logger.info("Full sync requested")
    try:
        result = await sync.sync_all()
        
        if result["curriculum"]["changed"]:
            validator.reload()
            logger.info("Curriculum reloaded")
        
        if result["content"]["changed"]:
            recommender.reload()
            logger.info("Recommender reloaded")
        
        return result
    except Exception as e:
        logger.error(f"Full sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
