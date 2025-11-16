"""
API Version 1 Blueprint

Modern Flask blueprint structure for the cyberbullying detection API.
"""

from flask import Blueprint

from .detection import detection_bp
from .management import management_bp
from .monitoring import monitoring_bp

# Create main API v1 blueprint
api_v1_bp = Blueprint('api_v1', __name__)

# Register sub-blueprints
api_v1_bp.register_blueprint(detection_bp, url_prefix='/detect')
api_v1_bp.register_blueprint(management_bp, url_prefix='/manage')
api_v1_bp.register_blueprint(monitoring_bp, url_prefix='/monitor')

# API v1 root endpoint
@api_v1_bp.route('/')
def api_info():
    """API version 1 information endpoint."""
    return {
        'version': '1.0.0',
        'name': 'Cyberbullying Detection API v1',
        'endpoints': {
            'detection': '/api/v1/detect',
            'management': '/api/v1/manage', 
            'monitoring': '/api/v1/monitor'
        },
        'documentation': '/docs'
    }
