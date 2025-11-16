"""
Adapter to integrate the multilingual detector with existing code
"""

import logging
from multilingual_detector import MultilingualBullyingDetector
from typing import Dict, List

logger = logging.getLogger(__name__)

class EnhancedIntelligentDetector:
    """
    Enhanced detector that combines the existing intelligent detector
    with the new multilingual capabilities
    """
    
    def __init__(self, data_file_path: str = None, openai_api_key: str = None):
        """Initialize with multilingual support"""
        # Initialize multilingual detector for better phrase handling
        self.multilingual_detector = MultilingualBullyingDetector(data_file_path)
        
        # Store OpenAI key for future AI enhancements
        self.openai_api_key = openai_api_key
        
        # Compatibility attributes
        self.category_patterns = {}
        self._setup_compatibility_layer()
        
        logger.info("EnhancedIntelligentDetector initialized")
    
    def _setup_compatibility_layer(self):
        """Setup compatibility with existing code"""
        # Map language data to category patterns for compatibility
        for language, lang_data in self.multilingual_detector.language_data.items():
            if language == 'english':
                self.category_patterns['direct_insults'] = {
                    'keywords': lang_data['single_words'],
                    'multi_word_phrases': [p['phrase'] for p in lang_data['phrases']],
                    'patterns': lang_data['patterns']
                }
            elif language == 'hindi':
                self.category_patterns['hindi_profanity'] = {
                    'keywords': lang_data['single_words'],
                    'multi_word_phrases': [p['phrase'] for p in lang_data['phrases']],
                    'patterns': lang_data['patterns']
                }
    
    def detect_bullying_enhanced(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Enhanced detection method that uses multilingual detector
        
        This method maintains compatibility with the existing interface
        while using the improved multilingual detection
        """
        # Use multilingual detector
        result = self.multilingual_detector.detect_bullying(text, confidence_threshold)
        
        # Convert to expected format for compatibility
        detected_categories = []
        
        # Group detections by language/category
        language_groups = {}
        for item in result.get('detected_items', []):
            lang = item.get('language', 'unknown')
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(item['value'])
        
        # Create category entries
        for language, items in language_groups.items():
            category_name = 'direct_insults' if language == 'english' else f'{language}_profanity'
            detected_categories.append({
                'category': category_name,
                'items': items,
                'score': 0.5,  # Default score
                'severity': result.get('severity', 'medium')
            })
        
        # Build enhanced result
        enhanced_result = {
            'is_bullying': result['is_bullying'],
            'confidence': result['confidence'],
            'severity': result['severity'],
            'detected_categories': detected_categories,
            'risk_indicators': [],
            'sentiment_analysis': {'compound': -0.5 if result['is_bullying'] else 0.5},
            'context_analysis': {'score': 0.0, 'indicators': []},
            'intent_classification': {},
            'detection_method': 'enhanced_multilingual',
            'detected_languages': result.get('detected_languages', []),
            'detected_items': result.get('detected_items', [])
        }
        
        return enhanced_result
    
    def _check_phrase_sequence(self, text: str, phrase: str) -> bool:
        """
        Check if a phrase appears in sequence in the text
        Uses the multilingual detector's implementation
        """
        words = self.multilingual_detector._tokenize_text(text)
        phrase_data = {
            'phrase': phrase,
            'words': phrase.split(),
            'word_count': len(phrase.split())
        }
        found, _ = self.multilingual_detector._check_multi_word_phrase(words, phrase_data)
        return found
    
    def add_bullying_words(self, words: List[str], language: str = 'english') -> int:
        """Add new words to the database"""
        # Determine if words are single or phrases
        single_words = []
        phrases = []
        
        for word in words:
            if ' ' in word.strip():
                phrases.append(word)
            else:
                single_words.append(word)
        
        added = 0
        if single_words:
            added += self.multilingual_detector.add_words(single_words, language, 'single')
        if phrases:
            added += self.multilingual_detector.add_words(phrases, language, 'phrase')
        
        # Update compatibility layer
        self._setup_compatibility_layer()
        
        return added


def update_existing_detectors():
    """
    Function to update existing detector imports in other files
    This can be used to patch existing code
    """
    import sys
    
    # Monkey patch the IntelligentBullyingDetector if it's imported
    if 'intelligent_detector' in sys.modules:
        sys.modules['intelligent_detector'].IntelligentBullyingDetector = EnhancedIntelligentDetector
        logger.info("Patched IntelligentBullyingDetector with enhanced version")
    
    return True
