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

load_dotenv()

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

BACKEND_URL = os.getenv("BACKEND_URL", "https://hanzimaster-backend-v2.isak-parild.workers.dev")
DATA_DIR = os.getenv("DATA_DIR", "./data")
API_KEY = os.getenv("VALIDATOR_API_KEY", "")  # Required for /sync
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Allowed origins - locked down for production
ALLOWED_ORIGINS = [
    BACKEND_URL,
    "https://hanzimaster-backend-v2.isak-parild.workers.dev",
]

# Add localhost for development
if ENVIRONMENT == "development":
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8787",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ])

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
    return {
        "version": curriculum["version"],
        "word_count": curriculum["word_count"],
        "loaded": curriculum["loaded"]
    }
