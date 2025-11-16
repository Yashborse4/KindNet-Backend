"""
Pydantic Models for Request/Response Validation

Modern data validation using Pydantic for type safety and automatic validation.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


class AnalysisMode(str, Enum):
    """Analysis mode options."""
    STANDARD = "standard"
    STRICT = "strict" 
    LENIENT = "lenient"


class SeverityLevel(str, Enum):
    """Severity level options."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionRequest(BaseModel):
    """Single text detection request."""
    
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze for cyberbullying")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Confidence threshold for detection")
    include_details: bool = Field(True, description="Include detailed analysis in response")
    analysis_mode: AnalysisMode = Field(AnalysisMode.STANDARD, description="Analysis mode")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    @validator('text')
    def validate_text_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class BatchDetectionRequest(BaseModel):
    """Batch text detection request."""
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Confidence threshold for detection")
    include_details: bool = Field(True, description="Include detailed analysis in response")
    analysis_mode: AnalysisMode = Field(AnalysisMode.STANDARD, description="Analysis mode")
    parallel_processing: bool = Field(True, description="Enable parallel processing for better performance")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and validate length
        valid_texts = []
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            
            text = text.strip()
            if text:
                if len(text) > 5000:
                    raise ValueError(f"Text at index {i} exceeds maximum length of 5000 characters")
                valid_texts.append(text)
        
        if not valid_texts:
            raise ValueError("No valid texts found after filtering empty strings")
        
        return valid_texts


class AddWordsRequest(BaseModel):
    """Request to add new bullying words to database."""
    
    words: List[str] = Field(..., min_items=1, max_items=50, description="List of words to add")
    category: Optional[str] = Field("direct_insults", description="Category for the words")
    severity: SeverityLevel = Field(SeverityLevel.MEDIUM, description="Severity level for the words")
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    @validator('words')
    def validate_words(cls, v):
        if not v:
            raise ValueError("Words list cannot be empty")
        
        valid_words = []
        for word in v:
            if isinstance(word, str) and word.strip():
                clean_word = word.strip().lower()
                if len(clean_word) > 0 and len(clean_word) <= 50:
                    valid_words.append(clean_word)
        
        if not valid_words:
            raise ValueError("No valid words provided")
        
        return valid_words


class DetectionResponse(BaseModel):
    """Single text detection response."""
    
    is_bullying: bool = Field(..., description="Whether cyberbullying was detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    severity: SeverityLevel = Field(..., description="Severity level if bullying detected")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    detected_categories: Optional[List[Dict[str, Any]]] = Field(None, description="Categories detected")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    detection_method: Optional[str] = Field(None, description="Detection method used")


class BatchDetectionResponse(BaseModel):
    """Batch detection response."""
    
    results: List[DetectionResponse] = Field(..., description="Individual detection results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")


class TextValidationResponse(BaseModel):
    """Text validation response."""
    
    is_valid: bool = Field(..., description="Whether text is valid for analysis")
    text_length: int = Field(..., description="Text length in characters")
    word_count: int = Field(..., description="Word count")
    language: Optional[str] = Field(None, description="Detected language")
    encoding: Optional[str] = Field(None, description="Text encoding")
    issues: Optional[List[str]] = Field(None, description="Validation issues if any")


class StatisticsResponse(BaseModel):
    """Statistics response."""
    
    total_detections: int = Field(..., description="Total number of detections performed")
    bullying_detected: int = Field(..., description="Number of bullying instances detected")
    accuracy_rate: Optional[float] = Field(None, description="Detection accuracy rate")
    category_breakdown: Optional[Dict[str, int]] = Field(None, description="Breakdown by category")
    severity_breakdown: Optional[Dict[str, int]] = Field(None, description="Breakdown by severity")
    processing_stats: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency status")
    timestamp: str = Field(..., description="Response timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Generic success response model."""
    
    success: bool = Field(True, description="Success indicator")
    message: Optional[str] = Field(None, description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: str = Field(..., description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
