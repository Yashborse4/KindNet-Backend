#!/usr/bin/env python3
"""
Cyberbullying Detection API - Application Entry Point

Modern Flask application entry point using the factory pattern.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from app import create_app
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main application entry point."""
    
    # Get environment
    environment = os.environ.get('FLASK_ENV', 'development')
    
    # Create application
    app = create_app(environment)
    
    # Get configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Log startup information
    logger.info(f"Starting Cyberbullying Detection API")
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    
    try:
        # Run the application
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
