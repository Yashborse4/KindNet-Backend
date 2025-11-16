"""
Cyberbullying Detection API

A modern Flask application for detecting cyberbullying in text content
using AI and machine learning techniques.
"""

import os
import logging
from typing import Dict, Any, Optional
from flask import Flask
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from app.utils.config_manager import ConfigManager
from app.utils.logger import setup_logging
from app.api.v1 import api_v1_bp
from app.utils.error_handlers import register_error_handlers
from app.utils.middleware import setup_middleware
from app.services.monitoring import setup_monitoring

# Global config manager instance
config_manager: Optional[ConfigManager] = None

def create_app(config_name: str = None) -> Flask:
    """
    Application factory function that creates and configures Flask app.
    
    Args:
        config_name: Environment name (development, production, testing)
        
    Returns:
        Configured Flask application instance
    """
    global config_manager
    
    # Determine environment
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    # Create Flask instance
    app = Flask(__name__)
    
    # Initialize configuration manager
    config_manager = ConfigManager(config_name)
    
    # Configure app from environment and config files
    configure_app(app, config_manager)
    
    # Setup logging
    setup_logging(app, config_manager)
    
    # Register blueprints
    register_blueprints(app)
    
    # Setup middleware
    setup_middleware(app, config_manager)
    
    # Register error handlers
    register_error_handlers(app, config_manager)
    
    # Setup monitoring
    setup_monitoring(app, config_manager)
    
    # Setup CORS
    setup_cors(app, config_manager)
    
    app.logger.info(f"Application created in {config_name} mode")
    
    return app


def configure_app(app: Flask, config_manager: ConfigManager) -> None:
    """Configure Flask app with settings from config manager."""
    
    # Basic Flask configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = config_manager.get('debug', False)
    app.config['TESTING'] = config_manager.get('testing', False)
    
    # Security settings
    app.config['MAX_CONTENT_LENGTH'] = config_manager.get('api.validation.max_content_length', 16 * 1024)
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = config_manager.get('debug', False)
    
    # OpenAI configuration
    app.config['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
    app.config['OPENAI_MODEL'] = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
    app.config['OPENAI_MAX_TOKENS'] = int(os.environ.get('OPENAI_MAX_TOKENS', 300))
    app.config['OPENAI_TEMPERATURE'] = float(os.environ.get('OPENAI_TEMPERATURE', 0.3))
    
    # Detection configuration  
    app.config['CONFIDENCE_THRESHOLD'] = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.7))
    app.config['LOCAL_DATABASE_PATH'] = os.environ.get('LOCAL_DATABASE_PATH', 'data/bullying_words.json')
    
    # Rate limiting
    rate_limit_config = config_manager.get('api.rate_limiting', {})
    app.config['RATE_LIMITING_ENABLED'] = rate_limit_config.get('enabled', False)
    app.config['RATE_LIMIT_PER_MINUTE'] = rate_limit_config.get('requests_per_minute', 60)
    app.config['RATE_LIMIT_BURST'] = rate_limit_config.get('burst_limit', 10)
    
    # Caching
    cache_config = config_manager.get('api.caching', {})
    app.config['CACHING_ENABLED'] = cache_config.get('enabled', True)
    app.config['CACHE_TTL'] = cache_config.get('ttl_seconds', 3600)
    app.config['CACHE_MAX_SIZE'] = cache_config.get('max_cache_size', 1000)


def register_blueprints(app: Flask) -> None:
    """Register Flask blueprints."""
    
    # Register API v1 blueprint
    app.register_blueprint(api_v1_bp, url_prefix='/api/v1')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Simple health check endpoint."""
        return {
            'status': 'healthy',
            'service': 'Cyberbullying Detection API',
            'version': '2.0.0',
            'environment': config_manager.environment if config_manager else 'unknown'
        }


def setup_cors(app: Flask, config_manager: ConfigManager) -> None:
    """Setup CORS configuration."""
    
    cors_config = config_manager.get('api.cors', {})
    
    CORS(
        app,
        origins=cors_config.get('origins', ['*']),
        methods=cors_config.get('methods', ['GET', 'POST']),
        allow_headers=cors_config.get('allow_headers', ['Content-Type']),
        supports_credentials=cors_config.get('supports_credentials', False)
    )


def get_config_manager() -> Optional[ConfigManager]:
    """Get the global config manager instance."""
    return config_manager


# Version info
__version__ = "2.0.0"
__author__ = "Cyberbullying Detection Team"
__email__ = "support@cyberbullying-detection.com"
