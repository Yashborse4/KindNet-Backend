"""
Detection API Endpoints

Modern Flask blueprint for cyberbullying detection endpoints with:
- Pydantic validation
- Comprehensive error handling
- Rate limiting
- Caching
- Detailed logging
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Blueprint, request, current_app
from pydantic import BaseModel, Field, ValidationError
from werkzeug.exceptions import BadRequest

from app.services.detection_service import DetectionService
from app.utils.response_builder import ResponseBuilder
from app.utils.rate_limiter import RateLimiter
from app.utils.cache import CacheManager
from app.utils.logger import get_logger
from app.models.schemas import DetectionRequest, BatchDetectionRequest, DetectionResponse

logger = get_logger(__name__)

# Create blueprint
detection_bp = Blueprint('detection', __name__)

# Initialize services (will be properly injected in production)
detection_service = None
rate_limiter = None
cache_manager = None
response_builder = ResponseBuilder()


def init_detection_services(app):
    """Initialize detection services with app context."""
    global detection_service, rate_limiter, cache_manager
    
    with app.app_context():
        detection_service = DetectionService()
        rate_limiter = RateLimiter()
        cache_manager = CacheManager()


@detection_bp.before_request
def before_request():
    """Pre-request processing."""
    # Initialize services if not done
    if detection_service is None:
        init_detection_services(current_app)
    
    # Rate limiting
    if current_app.config.get('RATE_LIMITING_ENABLED', False):
        if not rate_limiter.is_allowed(request.remote_addr):
            return response_builder.rate_limit_exceeded()
    
    # Log request
    logger.info(
        "Detection API request received",
        extra={
            'endpoint': request.endpoint,
            'method': request.method,
            'remote_addr': request.remote_addr,
            'content_length': request.content_length
        }
    )


@detection_bp.route('/', methods=['POST'])
def detect_single():
    """
    Detect cyberbullying in a single text.
    
    Request body:
    {
        "text": "Text to analyze",
        "confidence_threshold": 0.7,  // optional
        "include_details": true,      // optional
        "analysis_mode": "standard"   // optional: standard, strict, lenient
    }
    """
    try:
        # Validate request format
        if not request.is_json:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        data = request.get_json()
        if not data:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        # Validate with Pydantic
        try:
            detection_request = DetectionRequest(**data)
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return response_builder.validation_error(e)
        
        # Check cache
        cache_key = None
        if current_app.config.get('CACHING_ENABLED', True):
            cache_key = f"detect:{hash(detection_request.text)}"
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for detection request")
                return response_builder.success(cached_result, 'api.responses.success.detection_complete')
        
        # Perform detection
        start_time = datetime.utcnow()
        
        result = detection_service.detect_bullying(
            text=detection_request.text,
            confidence_threshold=detection_request.confidence_threshold,
            include_details=detection_request.include_details,
            analysis_mode=detection_request.analysis_mode
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        result['processing_time'] = processing_time
        
        # Cache result if enabled
        if cache_key and current_app.config.get('CACHING_ENABLED', True):
            cache_manager.set(cache_key, result, ttl=current_app.config.get('CACHE_TTL', 3600))
        
        # Log result
        logger.info(
            "Detection completed",
            extra={
                'is_bullying': result.get('is_bullying', False),
                'confidence': result.get('confidence', 0),
                'processing_time': processing_time,
                'text_length': len(detection_request.text),
                'cache_used': cache_key is not None
            }
        )
        
        return response_builder.success(result, 'api.responses.success.detection_complete')
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return response_builder.error('api.responses.errors.internal_server_error', 500)


@detection_bp.route('/batch', methods=['POST'])
def detect_batch():
    """
    Detect cyberbullying in multiple texts.
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "confidence_threshold": 0.7,  // optional
        "include_details": true,      // optional
        "analysis_mode": "standard",  // optional
        "parallel_processing": true   // optional
    }
    """
    try:
        # Validate request format
        if not request.is_json:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        data = request.get_json()
        if not data:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        # Validate with Pydantic
        try:
            batch_request = BatchDetectionRequest(**data)
        except ValidationError as e:
            logger.warning(f"Batch validation error: {e}")
            return response_builder.validation_error(e)
        
        # Check batch size limits
        max_batch_size = current_app.config.get('MAX_BATCH_SIZE', 100)
        if len(batch_request.texts) > max_batch_size:
            return response_builder.error(
                'api.responses.errors.batch_size_exceeded',
                400,
                details={'max_size': max_batch_size, 'requested_size': len(batch_request.texts)}
            )
        
        # Log batch processing start
        logger.info(
            f"Starting batch detection for {len(batch_request.texts)} texts",
            extra={
                'batch_size': len(batch_request.texts),
                'parallel_processing': batch_request.parallel_processing,
                'analysis_mode': batch_request.analysis_mode
            }
        )
        
        # Perform batch detection
        start_time = datetime.utcnow()
        
        results = detection_service.detect_bullying_batch(
            texts=batch_request.texts,
            confidence_threshold=batch_request.confidence_threshold,
            include_details=batch_request.include_details,
            analysis_mode=batch_request.analysis_mode,
            parallel_processing=batch_request.parallel_processing
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Compile batch statistics
        bullying_count = sum(1 for r in results if r.get('is_bullying', False))
        avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results) if results else 0
        
        batch_result = {
            'results': results,
            'summary': {
                'total_processed': len(results),
                'bullying_detected': bullying_count,
                'average_confidence': round(avg_confidence, 3),
                'processing_time': processing_time,
                'texts_per_second': round(len(results) / processing_time, 2) if processing_time > 0 else 0
            }
        }
        
        # Log batch completion
        logger.info(
            "Batch detection completed",
            extra={
                'total_processed': len(results),
                'bullying_detected': bullying_count,
                'processing_time': processing_time,
                'avg_confidence': avg_confidence
            }
        )
        
        return response_builder.success(batch_result, 'api.responses.success.batch_complete')
        
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}", exc_info=True)
        return response_builder.error('api.responses.errors.internal_server_error', 500)


@detection_bp.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Detailed text analysis with breakdown of detection factors.
    
    Returns comprehensive analysis including:
    - Sentiment analysis
    - Context indicators
    - Pattern matches
    - Confidence breakdown
    """
    try:
        if not request.is_json:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        data = request.get_json()
        if not data or 'text' not in data:
            return response_builder.error('api.responses.errors.missing_text_field', 400)
        
        text = data.get('text', '').strip()
        if not text:
            return response_builder.error('api.responses.errors.empty_text', 400)
        
        # Perform detailed analysis
        analysis = detection_service.analyze_text_detailed(text)
        
        logger.info(
            "Detailed analysis completed",
            extra={
                'text_length': len(text),
                'analysis_components': list(analysis.keys())
            }
        )
        
        return response_builder.success(analysis, 'api.responses.success.detection_complete')
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return response_builder.error('api.responses.errors.internal_server_error', 500)


@detection_bp.route('/validate', methods=['POST'])
def validate_text():
    """
    Validate text without performing full detection.
    
    Useful for checking text format, length, and basic characteristics.
    """
    try:
        if not request.is_json:
            return response_builder.error('api.responses.errors.invalid_content_type', 400)
        
        data = request.get_json()
        if not data or 'text' not in data:
            return response_builder.error('api.responses.errors.missing_text_field', 400)
        
        text = data.get('text', '')
        
        # Perform validation
        validation_result = detection_service.validate_text(text)
        
        return response_builder.success(validation_result, 'api.responses.success.detection_complete')
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return response_builder.error('api.responses.errors.internal_server_error', 500)
