# Cyberbullying Detection API - Modern Implementation Guide

## 🏗️ Architecture Overview

The modernized cyberbullying detection API follows industry best practices with:

- **Factory Pattern**: Application factory for flexible configuration
- **Blueprint Structure**: Modular API organization with versioning
- **Pydantic Validation**: Type-safe request/response validation
- **Configuration Management**: Environment-based config with JSON overrides
- **Structured Logging**: JSON logging with request tracing
- **Error Handling**: Comprehensive error handling with consistent responses
- **Testing**: Full test coverage with pytest and fixtures

## 🚀 Quick Start with Modern Structure

### 1. Run the Application

```bash
# Using the new entry point
python run.py

# Or with environment variables
FLASK_ENV=development DEBUG=true python run.py
```

### 2. API Base URL

```
http://localhost:5000/api/v1
```

## 📚 API Endpoints

### Detection Endpoints

#### Single Text Detection
```http
POST /api/v1/detect/
Content-Type: application/json

{
  "text": "Your text to analyze",
  "confidence_threshold": 0.7,
  "include_details": true,
  "analysis_mode": "standard"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_bullying": false,
    "confidence": 0.15,
    "severity": "none",
    "explanation": "No cyberbullying detected. The message appears to be safe.",
    "detected_categories": [],
    "processing_time": 0.052,
    "detection_method": "enhanced_local"
  },
  "timestamp": "2024-08-22T14:09:41.123Z"
}
```

#### Batch Detection
```http
POST /api/v1/detect/batch
Content-Type: application/json

{
  "texts": ["Hello world", "You are amazing", "Great job!"],
  "confidence_threshold": 0.7,
  "include_details": true,
  "parallel_processing": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "is_bullying": false,
        "confidence": 0.05,
        "severity": "none",
        "processing_time": 0.031
      }
    ],
    "summary": {
      "total_processed": 3,
      "bullying_detected": 0,
      "average_confidence": 0.067,
      "processing_time": 0.156,
      "texts_per_second": 19.23
    }
  },
  "timestamp": "2024-08-22T14:09:41.123Z"
}
```

#### Text Analysis (Detailed)
```http
POST /api/v1/detect/analyze
Content-Type: application/json

{
  "text": "Your text for detailed analysis"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sentiment_analysis": {
      "compound": 0.2,
      "pos": 0.3,
      "neu": 0.6,
      "neg": 0.1
    },
    "context_analysis": {
      "score": 0.1,
      "indicators": ["neutral_tone"]
    },
    "intent_classification": {
      "threat": 0.0,
      "insult": 0.0,
      "exclusion": 0.0,
      "harassment": 0.0
    },
    "detected_categories": [],
    "risk_indicators": []
  }
}
```

#### Text Validation
```http
POST /api/v1/detect/validate
Content-Type: application/json

{
  "text": "Text to validate"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "text_length": 16,
    "word_count": 3,
    "language": "en",
    "encoding": "utf-8",
    "issues": []
  }
}
```

### Management Endpoints

#### Add Bullying Words
```http
POST /api/v1/manage/words
Content-Type: application/json

{
  "words": ["newbadword1", "newbadword2"],
  "category": "direct_insults",
  "severity": "medium"
}
```

#### Get Statistics
```http
GET /api/v1/monitor/stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_detections": 1250,
    "bullying_detected": 125,
    "accuracy_rate": 0.92,
    "category_breakdown": {
      "direct_insults": 45,
      "threats": 12,
      "social_exclusion": 23
    },
    "processing_stats": {
      "avg_response_time": 0.087,
      "cache_hit_rate": 0.34
    }
  }
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Cyberbullying Detection API",
  "version": "2.0.0",
  "environment": "development",
  "timestamp": "2024-08-22T14:09:41.123Z"
}
```

## 🔧 Configuration

### Environment Variables

```bash
# Application Settings
FLASK_ENV=development
DEBUG=true
HOST=0.0.0.0
PORT=5000
SECRET_KEY=your-secret-key

# AI Services
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Detection Settings
CONFIDENCE_THRESHOLD=0.7
LOCAL_DATABASE_PATH=data/bullying_words.json

# Performance
RATE_LIMIT_ENABLED=true
CACHE_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Configuration Files

The modern structure uses JSON configuration files:

- `config/settings.json` - Base application settings
- `config/messages.json` - All user-facing messages
- `config/environments/development.json` - Development overrides
- `config/environments/production.json` - Production overrides

Example `config/settings.json`:
```json
{
  "detection": {
    "thresholds": {
      "confidence": {
        "default": 0.7,
        "strict": 0.9,
        "lenient": 0.5
      }
    },
    "features": {
      "enable_sentiment_analysis": true,
      "enable_context_analysis": true,
      "enable_openai_fallback": true
    }
  },
  "api": {
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60
    },
    "caching": {
      "enabled": true,
      "ttl_seconds": 3600
    }
  }
}
```

## 🛠️ Modern Features

### 1. Pydantic Validation

All requests are validated using Pydantic models:

```python
from pydantic import BaseModel, Field

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    include_details: bool = Field(True)
    analysis_mode: Literal["standard", "strict", "lenient"] = Field("standard")
```

### 2. Response Builder

Consistent API responses:

```python
# Success response
response_builder.success(data, 'api.responses.success.detection_complete')

# Error response
response_builder.error('api.responses.errors.invalid_text', 400)
```

### 3. Configuration Management

Dynamic configuration loading:

```python
config_manager = ConfigManager('development')
threshold = config_manager.get_threshold('confidence', 'default')
message = config_manager.get_message('api.responses.success.detection_complete')
```

### 4. Structured Logging

JSON logging with request context:

```python
logger.info(
    "Detection completed",
    extra={
        'is_bullying': result['is_bullying'],
        'confidence': result['confidence'],
        'processing_time': processing_time
    }
)
```

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific categories
pytest -m unit
pytest -m integration
pytest -m api
```

### Test Structure

```
tests/
├── conftest.py           # Test configuration and fixtures
├── unit/                 # Unit tests
│   ├── test_config.py
│   ├── test_detection.py
│   └── test_utils.py
├── integration/          # Integration tests
│   └── test_services.py
└── api/                  # API tests
    ├── test_detection_endpoints.py
    └── test_health_endpoints.py
```

## 🐳 Docker Deployment

### Development

```bash
# Start with Docker Compose
docker-compose up --build

# Services:
# - API: http://localhost:5000
# - Redis: localhost:6379
```

### Production

```bash
# Build production image
docker build -t cyberbullying-api .

# Run with production settings
docker run -e FLASK_ENV=production \
           -e OPENAI_API_KEY=your-key \
           -p 5000:5000 \
           cyberbullying-api
```

### With Monitoring

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Additional services:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3001 (admin/admin123)
```

## 📊 Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": "Text field cannot be empty",
  "error_code": "VALIDATION_ERROR",
  "details": {
    "validation_errors": [
      {
        "field": "text",
        "message": "Field required",
        "type": "missing"
      }
    ]
  },
  "timestamp": "2024-08-22T14:09:41.123Z"
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (validation errors)
- `404` - Not Found
- `405` - Method Not Allowed
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error
- `503` - Service Unavailable

## 🚀 Performance Features

### Caching

```python
# Cache detection results
cache_key = f"detect:{hash(text)}"
cached_result = cache_manager.get(cache_key)
if cached_result:
    return cached_result
```

### Rate Limiting

```python
# Per-IP rate limiting
if not rate_limiter.is_allowed(request.remote_addr):
    return response_builder.rate_limit_exceeded()
```

### Async Processing

```python
# Batch processing with parallel execution
results = await asyncio.gather(*[
    detect_single(text) for text in texts
])
```

## 🔍 Monitoring & Observability

### Metrics Endpoints

- `GET /api/v1/monitor/metrics` - Prometheus metrics
- `GET /api/v1/monitor/health` - Detailed health check
- `GET /api/v1/monitor/stats` - Application statistics

### Logging

```json
{
  "timestamp": "2024-08-22T14:09:41.123Z",
  "level": "INFO",
  "logger": "app.api.v1.detection",
  "message": "Detection completed",
  "request": {
    "method": "POST",
    "path": "/api/v1/detect/",
    "remote_addr": "127.0.0.1"
  },
  "extra": {
    "is_bullying": false,
    "confidence": 0.15,
    "processing_time": 0.052
  }
}
```

## 🔐 Security Features

- Input validation and sanitization
- Rate limiting protection
- CORS configuration
- Environment-based secrets
- Request size limits
- Secure headers in production

## 📈 Migration from Legacy Code

The new structure provides these improvements over the original:

1. **Better Organization**: Proper package structure with blueprints
2. **Type Safety**: Pydantic validation prevents runtime errors
3. **Configuration**: Centralized, environment-aware configuration
4. **Testing**: Comprehensive test suite with fixtures
5. **Monitoring**: Built-in metrics and health checks
6. **Documentation**: Auto-generated API documentation
7. **Deployment**: Docker and Docker Compose support
8. **Error Handling**: Consistent error responses
9. **Logging**: Structured logging with request tracing
10. **Performance**: Caching, rate limiting, and async support
