"""
Test Configuration and Fixtures

Shared pytest fixtures and configuration for the test suite.
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from app.utils.config_manager import ConfigManager


@pytest.fixture(scope="session")
def app_config():
    """Create test configuration."""
    return {
        'TESTING': True,
        'DEBUG': True,
        'SECRET_KEY': 'test-secret-key',
        'WTF_CSRF_ENABLED': False,
        'OPENAI_API_KEY': 'test-openai-key',
        'CONFIDENCE_THRESHOLD': 0.7,
        'RATE_LIMITING_ENABLED': False,
        'CACHING_ENABLED': False,
    }


@pytest.fixture(scope="session")
def app(app_config):
    """Create Flask application for testing."""
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['LOCAL_DATABASE_PATH'] = os.path.join(temp_dir, 'test_bullying_words.json')
        
        # Mock the config manager to use test environment
        with patch('app.utils.config_manager.ConfigManager') as mock_config_manager:
            mock_instance = Mock()
            mock_instance.environment = 'testing'
            mock_instance.get.return_value = None
            mock_instance.get_message.return_value = "Test message"
            mock_instance.is_feature_enabled.return_value = True
            mock_instance.get_threshold.return_value = 0.7
            mock_config_manager.return_value = mock_instance
            
            app = create_app('testing')
            app.config.update(app_config)
            
            yield app


@pytest.fixture(scope="function")
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture(scope="function")
def runner(app):
    """Create test CLI runner."""
    return app.test_cli_runner()


@pytest.fixture(scope="function")
def app_context(app):
    """Create application context."""
    with app.app_context():
        yield app


@pytest.fixture(scope="function")
def request_context(app):
    """Create request context."""
    with app.test_request_context():
        yield app


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        'safe': [
            "Hello, how are you today?",
            "The weather is nice.",
            "I love programming.",
            "Good morning everyone!",
            "Have a great day!"
        ],
        'bullying': [
            "You are stupid and worthless",
            "Nobody likes you, loser",
            "Kill yourself, idiot",
            "You're ugly and fat",
            "Go away, freak"
        ],
        'edge_cases': [
            "",  # Empty string
            " ",  # Whitespace only
            "a" * 5001,  # Too long
            "123456789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
        ]
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('openai.OpenAI') as mock_client:
        mock_instance = Mock()
        
        # Mock chat completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
            "is_bullying": true,
            "confidence": 0.85,
            "severity": "high",
            "primary_category": "direct_insults",
            "detected_patterns": ["insults"],
            "risk_factors": ["aggressive_language"],
            "reasoning": "Contains direct insults and aggressive language",
            "recommended_action": "warn"
        }
        '''
        
        mock_instance.chat.completions.create.return_value = mock_response
        mock_client.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_detection_service():
    """Mock detection service for testing."""
    with patch('app.services.detection_service.DetectionService') as mock_service:
        mock_instance = Mock()
        
        # Mock detection results
        mock_instance.detect_bullying.return_value = {
            'is_bullying': False,
            'confidence': 0.1,
            'severity': 'none',
            'detected_categories': [],
            'processing_time': 0.05
        }
        
        mock_instance.detect_bullying_batch.return_value = [
            {
                'is_bullying': False,
                'confidence': 0.1,
                'severity': 'none',
                'detected_categories': [],
                'processing_time': 0.05
            }
        ]
        
        mock_instance.validate_text.return_value = {
            'is_valid': True,
            'text_length': 10,
            'word_count': 2,
            'issues': []
        }
        
        mock_service.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    with patch('app.utils.rate_limiter.RateLimiter') as mock_limiter:
        mock_instance = Mock()
        mock_instance.is_allowed.return_value = True
        mock_limiter.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    with patch('app.utils.cache.CacheManager') as mock_cache:
        mock_instance = Mock()
        mock_instance.get.return_value = None  # Cache miss by default
        mock_instance.set.return_value = True
        mock_cache.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_openai: Tests requiring OpenAI API")
    config.addinivalue_line("markers", "requires_redis: Tests requiring Redis")


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip tests based on markers and environment
    if "requires_openai" in item.keywords:
        if not os.environ.get('OPENAI_API_KEY'):
            pytest.skip("OpenAI API key not available")
    
    if "requires_redis" in item.keywords:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
        except:
            pytest.skip("Redis not available")


# Custom assertions
def assert_valid_response(response):
    """Assert that a response is valid."""
    assert response.status_code < 500
    assert response.is_json
    data = response.get_json()
    assert 'success' in data
    assert 'timestamp' in data


def assert_error_response(response, expected_status=400):
    """Assert that a response is a valid error response."""
    assert response.status_code == expected_status
    assert response.is_json
    data = response.get_json()
    assert data['success'] is False
    assert 'error' in data
    assert 'timestamp' in data


def assert_success_response(response, expected_status=200):
    """Assert that a response is a valid success response."""
    assert response.status_code == expected_status
    assert response.is_json
    data = response.get_json()
    assert data['success'] is True
    assert 'data' in data
    assert 'timestamp' in data
