# Vocab Validator Service

A FastAPI service that validates AI-generated Chinese text against the HanziMaster curriculum.

## Purpose

When AI generates lesson content, this service checks that:
- Words are appropriate for the user's level (HSK + lesson)
- Target vocabulary is included
- No "forbidden" words (too advanced) slip through

## Security

- **CORS**: Locked to known origins (backend URL + localhost in dev)
- **API Key**: `/sync` endpoint requires `X-API-Key` header
- **Auto-sync**: Downloads curriculum on startup if cache missing
- **Retry logic**: Network calls retry 3x with exponential backoff

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BACKEND_URL` | No | `https://hanzimaster-backend-v2...` | Backend API URL |
| `DATA_DIR` | No | `./data` | Directory for curriculum cache |
| `VALIDATOR_API_KEY` | **Yes (prod)** | - | API key for `/sync` endpoint |
| `ENVIRONMENT` | No | `development` | `development` or `production` |

## API Endpoints

### `GET /health`
Health check (public)
```json
{
  "status": "healthy",
  "curriculum_loaded": true,
  "word_count": 11000,
  "version": "abc123",
  "environment": "production"
}
```

### `POST /validate`
Validate text (public)
```json
// Request
{
  "text": "你好，我学习中文。",
  "user_position": { "hsk": 1, "lesson": 3 },
  "target_words": ["学习"]
}

// Response
{
  "valid": true,
  "words_found": ["你好", "我", "学习", "中文"],
  "safe_words": ["你好", "我"],
  "target_words": ["学习"],
  "forbidden_words": [],
  "unknown_words": ["中文"],
  "stats": {
    "total_words": 4,
    "unique_words": 4,
    "safe_percentage": 50.0
  }
}
```

### `GET /get-vocabulary`
Get allowed words up to a lesson number (public) - **Used by AI Tutor generator**
```bash
curl "https://your-validator.sevalla.app/get-vocabulary?max_lesson=15"
```
Response:
```json
{
  "words": ["你", "好", "我", "是", "学习", ...],
  "count": 150,
  "max_lesson": 15
}
```

### `POST /sync`
Sync curriculum from backend (**requires API key**)
```bash
curl -X POST https://your-validator.sevalla.app/sync \
  -H "X-API-Key: your-secret-key"
```

### `GET /version`
Get curriculum version (public)

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run server
uvicorn app.main:app --reload --port 8000
```

## Deployment (Sevalla)

1. **Push code to GitHub**

2. **Create Sevalla app**
   - Type: Dockerfile
   - Port: 8000

3. **Set environment variables**
   ```
   BACKEND_URL=https://hanzimaster-backend-v2.isak-parild.workers.dev
   VALIDATOR_API_KEY=<generate-secure-key>
   ENVIRONMENT=production
   DATA_DIR=/app/data
   ```

4. **Deploy**

5. **Initialize curriculum**
   ```bash
   curl -X POST https://your-app.sevalla.app/sync \
     -H "X-API-Key: your-secret-key"
   ```

## Test Coverage

Run with coverage:
```bash
pytest tests/ -v --cov=app --cov-report=html
```

Current: **53+ tests** covering:
- Validator logic (27 tests)
- API endpoints (26 tests)
- Security (4 tests)
- Get Vocabulary (10 tests)

## Architecture

```
┌─────────────────┐     POST /validate     ┌─────────────────┐
│                 │ ───────────────────────▶│                 │
│  CF Worker      │                        │  Vocab          │
│  (Backend)      │◀─────────────────────── │  Validator      │
│                 │     {valid: true/false} │  (Python)       │
└─────────────────┘                        └─────────────────┘
        │                                          │
        │ POST /sync (daily)                       │
        └──────────────────────────────────────────┘
```

## Validation Logic

```
User Position: HSK 1, Lesson 3

Words from HSK 1, L1-3  →  SAFE ✓
Words at HSK 1, L3       →  TARGET 🎯
Words from HSK 1, L4+    →  FORBIDDEN ✗
Words from HSK 2+        →  FORBIDDEN ✗
Unknown words            →  FLAGGED (but don't fail)
```
