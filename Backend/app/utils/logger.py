"""
Logging Configuration

Modern structured logging setup with support for:
- JSON formatting for production
- Console and file output
- Log rotation
- Different log levels per environment
"""

import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, Any, Optional
from flask import Flask, has_request_context, request
from datetime import datetime

from app.utils.config_manager import ConfigManager


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs JSON structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log structure
        log_obj = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add request context if available
        if has_request_context():
            try:
                log_obj['request'] = {
                    'method': request.method,
                    'path': request.path,
                    'remote_addr': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', ''),
                }
            except Exception:
                # Ignore context errors
                pass
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in log_obj and not key.startswith('_') and key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage'
            ]:
                log_obj['extra'] = log_obj.get('extra', {})
                log_obj['extra'][key] = value
        
        return json.dumps(log_obj, ensure_ascii=False)


class DetailedFormatter(logging.Formatter):
    """Detailed human-readable formatter for development."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)-20s - %(levelname)-8s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(app: Flask, config_manager: ConfigManager) -> None:
    """
    Setup comprehensive logging for the application.
    
    Args:
        app: Flask application instance
        config_manager: Configuration manager instance
    """
    
    # Get logging configuration
    log_config = config_manager.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', 'detailed')
    console_output = log_config.get('console_output', True)
    file_output = log_config.get('file_output', True)
    
    # Set root logger level
    logging.root.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        if log_format == 'json':
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(DetailedFormatter())
        
        handlers.append(console_handler)
    
    # File handler with rotation
    if file_output:
        log_dir = app.instance_path.replace('instance', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'cyberbullying_detection.log')
        
        # Rotating file handler
        max_size = _parse_size(log_config.get('max_file_size', '10MB'))
        backup_count = log_config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        
        if log_format == 'json':
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(DetailedFormatter())
        
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=handlers,
        force=True
    )
    
    # Set specific logger levels
    _configure_specific_loggers(config_manager)
    
    # Setup Flask request logging
    _setup_request_logging(app, config_manager)
    
    app.logger.info(f"Logging configured with level {log_level} and format {log_format}")


def _parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., '10MB') to bytes.
    
    Args:
        size_str: Size string like '10MB', '1GB', etc.
        
    Returns:
        Size in bytes
    """
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }
    
    size_str = size_str.upper().strip()
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                number = float(size_str[:-len(unit)])
                return int(number * multiplier)
            except ValueError:
                pass
    
    # Default to bytes if no unit specified
    try:
        return int(size_str)
    except ValueError:
        return 10 * 1024 * 1024  # 10MB default


def _configure_specific_loggers(config_manager: ConfigManager) -> None:
    """Configure specific third-party loggers."""
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Set application loggers to appropriate levels
    app_loggers = [
        'app',
        'app.services',
        'app.utils',
        'app.api'
    ]
    
    log_level = config_manager.get('logging.level', 'INFO').upper()
    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, log_level, logging.INFO))


def _setup_request_logging(app: Flask, config_manager: ConfigManager) -> None:
    """Setup request/response logging middleware."""
    
    if not config_manager.get('detection.features.enable_detailed_logging', True):
        return
    
    @app.before_request
    def log_request_info():
        """Log incoming request information."""
        if request.endpoint not in ['health_check']:  # Skip health checks
            app.logger.info(
                "Incoming request",
                extra={
                    'request_method': request.method,
                    'request_path': request.path,
                    'request_args': dict(request.args),
                    'remote_addr': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', ''),
                    'content_length': request.content_length
                }
            )
    
    @app.after_request
    def log_response_info(response):
        """Log outgoing response information."""
        if request.endpoint not in ['health_check']:  # Skip health checks
            app.logger.info(
                "Outgoing response",
                extra={
                    'response_status': response.status_code,
                    'response_size': len(response.get_data()),
                    'request_path': request.path,
                    'processing_time': getattr(request, 'processing_time', None)
                }
            )
        return response


class LoggerMixin:
    """Mixin class to provide logger to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(func):
    """Decorator to log function performance."""
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.utcnow()
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(
                f"Function {func.__name__} completed",
                extra={
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'success'
                }
            )
            
            return result
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    'function': func.__name__,
                    'duration_seconds': duration,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise
    
    return wrapper
