from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openai
from werkzeug.exceptions import BadRequest

# Local imports
from config import Config
from detector_adapter import EnhancedIntelligentDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bullying_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize intelligent bullying detector (multilingual-aware)
detector = EnhancedIntelligentDetector()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Cyberbullying Detection API',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect_bullying():
    """
    Main endpoint to detect bullying in text
    Follows the detection hierarchy: Local DB -> OpenAI -> Final decision
    """
    try:
        # Validate request
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'text' not in data:
            raise BadRequest("Missing 'text' field in request body")
        
        text = data.get('text', '').strip()
        if not text:
            raise BadRequest("Text field cannot be empty")
        
        # Optional parameters
        include_details = data.get('include_details', True)
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        logger.info(f"Processing text: {text[:100]}...")  # Log first 100 chars
        
        # Perform enhanced bullying detection
        result = detector.detect_bullying_enhanced(
            text=text,
            confidence_threshold=confidence_threshold
        )
        
        # Log the result
        logger.info(f"Detection result: {result['is_bullying']} (confidence: {result['confidence']:.2f})")
        
        return jsonify({
            'success': True,
            'data': result,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect_bullying():
    """
    Endpoint to detect bullying in multiple texts
    """
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'texts' not in data:
            raise BadRequest("Missing 'texts' field in request body")
        
        texts = data.get('texts', [])
        if not isinstance(texts, list) or len(texts) == 0:
            raise BadRequest("'texts' must be a non-empty array")
        
        if len(texts) > 100:  # Limit batch size
            raise BadRequest("Maximum 100 texts allowed per batch")
        
        include_details = data.get('include_details', True)
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        logger.info(f"Processing batch of {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts):
            try:
                if not isinstance(text, str):
                    results.append({
                        'index': i,
                        'error': 'Text must be a string',
                        'is_bullying': False,
                        'confidence': 0.0
                    })
                    continue
                
                result = detector.detect_bullying_enhanced(
                    text=text.strip(),
                    confidence_threshold=confidence_threshold
                )
                result['index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing text {i}: {str(e)}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'is_bullying': False,
                    'confidence': 0.0
                })
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'total_processed': len(results),
                'bullying_detected': sum(1 for r in results if r.get('is_bullying', False))
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    except Exception as e:
        logger.error(f"Unexpected error in batch detection: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get detection statistics
    """
    try:
        stats = detector.get_enhanced_statistics()
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/api/add-words', methods=['POST'])
def add_bullying_words():
    """
    Add new bullying words to the local database
    """
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")
        
        data = request.get_json()
        if not data or 'words' not in data:
            raise BadRequest("Missing 'words' field in request body")
        
        words = data.get('words', [])
        if not isinstance(words, list) or len(words) == 0:
            raise BadRequest("'words' must be a non-empty array")
        
        # Validate words
        valid_words = []
        for word in words:
            if isinstance(word, str) and word.strip():
                valid_words.append(word.strip().lower())
        
        if not valid_words:
            raise BadRequest("No valid words provided")
        
        # Add words to detector
        added_count = detector.add_bullying_words(valid_words)
        
        logger.info(f"Added {added_count} new bullying words")
        
        return jsonify({
            'success': True,
            'data': {
                'words_added': added_count,
                'total_words': len(valid_words)
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except BadRequest as e:
        logger.warning(f"Bad request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    except Exception as e:
        logger.error(f"Error adding words: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'timestamp': datetime.utcnow().isoformat()
    }), 405

if __name__ == '__main__':
    logger.info("Starting Cyberbullying Detection API...")
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', False)
    )
