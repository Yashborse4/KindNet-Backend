import json
import re
import os
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import string
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultilingualBullyingDetector:
    """
    Enhanced cyberbullying detection system with:
    1. Proper multi-word phrase tokenization
    2. Support for multiple languages (English, Hindi, Marathi)
    3. Context-aware detection
    4. Improved word combination handling
    """
    
    def __init__(self, data_file_path: str = None):
        """Initialize the multilingual detector"""
        self.data_file_path = data_file_path or os.path.join('data', 'bullying_words.json')
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'bullying_detected': 0,
            'languages_detected': defaultdict(int),
            'phrase_matches': 0,
            'word_matches': 0,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Load enhanced database
        self.bullying_data = self._load_multilingual_database()
        
        # Preprocess data for faster lookup
        self._preprocess_multilingual_data()
        
        logger.info("MultilingualBullyingDetector initialized successfully")
    
    def _load_multilingual_database(self) -> Dict:
        """Load multilingual bullying database"""
        try:
            if not os.path.exists(self.data_file_path):
                logger.warning(f"Database file not found: {self.data_file_path}")
                return self._get_default_multilingual_data()
            
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure multilingual structure
            if 'languages' not in data:
                data = self._convert_to_multilingual_format(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            return self._get_default_multilingual_data()
    
    def _get_default_multilingual_data(self) -> Dict:
        """Default multilingual data structure"""
        return {
            'languages': {
                'english': {
                    'single_words': ['stupid', 'idiot', 'loser', 'ugly', 'fat', 'die', 'kill'],
                    'multi_word_phrases': ['kill yourself', 'nobody likes you', 'go die', 'you should die', 'go kill yourself'],
                    'patterns': [r'\byou\s+are\s+(so\s+)?(stupid|ugly|worthless)'],
                    'severity_map': {
                        'kill yourself': 'high',
                        'go die': 'high',
                        'you should die': 'high',
                        'go kill yourself': 'high',
                        'stupid': 'medium',
                        'ugly': 'medium',
                        'die': 'low',
                        'kill': 'low'
                    }
                },
                'hindi': {
                    'single_words': ['chutiya', 'gandu', 'harami', 'kutte', 'nalayak', 'gaand'],
                    'multi_word_phrases': ['teri maa ki', 'ullu ka patha', 'gaand maaru', 'teri maa ka'],
                    'patterns': [r'\bteri\s+maa\s+ki\b', r'\bgaand\s+maa?r'],
                    'severity_map': {
                        'madarchod': 'high',
                        'chutiya': 'medium',
                        'nalayak': 'low',
                        'teri maa ki': 'high',
                        'ullu ka patha': 'medium'
                    }
                },
                'marathi': {
                    'single_words': ['bokya', 'gandya', 'lavdya', 'bhikari'],
                    'multi_word_phrases': ['aai ghalya', 'tujhi aai'],
                    'patterns': [r'\btujhi\s+aai\b'],
                    'severity_map': {
                        'aai ghalya': 'high',
                        'bokya': 'medium',
                        'lavdya': 'medium',
                        'gandya': 'medium'
                    }
                }
            },
            'metadata': {
                'version': '2.0',
                'last_updated': datetime.utcnow().isoformat()
            }
        }
    
    def _convert_to_multilingual_format(self, data: Dict) -> Dict:
        """Convert legacy format to multilingual format"""
        multilingual_data = self._get_default_multilingual_data()
        
        # Extract all words and phrases from legacy format
        all_words = []
        all_phrases = []
        
        # From bullying_words
        if 'bullying_words' in data:
            all_words.extend(data['bullying_words'])
        
        # From bullying_phrases
        if 'bullying_phrases' in data:
            all_phrases.extend(data['bullying_phrases'])
        
        # From categories
        if 'bullying_categories' in data:
            for category, cat_data in data['bullying_categories'].items():
                if 'keywords' in cat_data:
                    for keyword in cat_data['keywords']:
                        if ' ' in keyword:
                            all_phrases.append(keyword)
                        else:
                            all_words.append(keyword)
        
        # Categorize words by language
        hindi_words = []
        hindi_phrases = []
        english_words = []
        english_phrases = []
        
        # Hindi detection patterns
        hindi_patterns = [
            'chutiya', 'bhosdike', 'gandu', 'lund', 'gaand', 'randi', 'harami',
            'kutte', 'suar', 'nalayak', 'bevakuf', 'ullu', 'kaminey', 'madarchod',
            'bhenchod', 'chod', 'teri', 'maa', 'behen', 'baap', 'bewakoof'
        ]
        
        # Categorize single words
        for word in all_words:
            word_lower = word.lower().strip()
            if any(hindi_word in word_lower for hindi_word in hindi_patterns):
                hindi_words.append(word_lower)
            else:
                english_words.append(word_lower)
        
        # Categorize phrases
        for phrase in all_phrases:
            phrase_lower = phrase.lower().strip()
            if any(hindi_word in phrase_lower for hindi_word in hindi_patterns):
                hindi_phrases.append(phrase_lower)
            else:
                english_phrases.append(phrase_lower)
        
        # Update multilingual data
        multilingual_data['languages']['english']['single_words'] = list(set(english_words))
        multilingual_data['languages']['english']['multi_word_phrases'] = list(set(english_phrases))
        multilingual_data['languages']['hindi']['single_words'] = list(set(hindi_words))
        multilingual_data['languages']['hindi']['multi_word_phrases'] = list(set(hindi_phrases))
        
        return multilingual_data
    
    def _preprocess_multilingual_data(self):
        """Preprocess data for efficient multi-language lookup"""
        self.language_data = {}
        
        for language, lang_data in self.bullying_data.get('languages', {}).items():
            # Create sets for O(1) lookup
            single_words_set = set(word.lower() for word in lang_data.get('single_words', []))
            
            # Process multi-word phrases
            phrase_data = []
            for phrase in lang_data.get('multi_word_phrases', []):
                phrase_lower = phrase.lower()
                phrase_words = phrase_lower.split()
                phrase_data.append({
                    'phrase': phrase_lower,
                    'words': phrase_words,
                    'word_count': len(phrase_words),
                    'severity': lang_data.get('severity_map', {}).get(phrase_lower, 'medium')
                })
            
            # Sort phrases by word count (descending) for priority matching
            phrase_data.sort(key=lambda x: x['word_count'], reverse=True)
            
            # Compile regex patterns
            compiled_patterns = []
            for pattern in lang_data.get('patterns', []):
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in {language}: {pattern} - {str(e)}")
            
            self.language_data[language] = {
                'single_words': single_words_set,
                'phrases': phrase_data,
                'patterns': compiled_patterns,
                'severity_map': lang_data.get('severity_map', {})
            }
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text while preserving multi-word context"""
        # Normalize text
        text = text.lower().strip()
        
        # Remove excessive punctuation but keep word boundaries
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Split into words while preserving some punctuation
        words = re.findall(r'\b[\w]+\b', text)
        
        return words
    
    def _detect_language(self, text: str) -> List[str]:
        """Detect which languages are present in the text"""
        detected_languages = []
        text_lower = text.lower()
        words = text_lower.split()
        
        # Hindi/Devanagari script detection
        if re.search(r'[\u0900-\u097F]', text):
            detected_languages.append('hindi')
        
        # Marathi detection (check for specific words)
        marathi_words = ['bokya', 'gandya', 'lavdya', 'aai', 'tujhi', 'majhi', 'kay', 'nahi', 'ghalya']
        if any(word in words for word in marathi_words):
            if 'marathi' not in detected_languages:
                detected_languages.append('marathi')
        
        # Hindi romanized detection (expanded list)
        hindi_words = ['teri', 'maa', 'behen', 'chutiya', 'gandu', 'madarchod', 'bhenchod', 
                      'ullu', 'patha', 'gaand', 'lund', 'randi', 'harami', 'kutte', 
                      'bhosdike', 'chod', 'kaminey', 'bewakoof', 'bakchod']
        if any(word in words for word in hindi_words):
            if 'hindi' not in detected_languages:
                detected_languages.append('hindi')
        
        # Always check English as well (multilingual text support)
        detected_languages.append('english')
        
        return list(set(detected_languages))  # Remove duplicates
    
    def _check_multi_word_phrase(self, words: List[str], phrase_data: Dict) -> Tuple[bool, float]:
        """
        Check if a multi-word phrase exists in the word list
        Returns (is_found, confidence_boost)
        """
        phrase_words = phrase_data['words']
        phrase_len = len(phrase_words)
        
        # Try to find the phrase with flexible matching
        for i in range(len(words) - phrase_len + 1):
            matched = True
            matched_positions = []
            
            # Check if all words of the phrase appear in sequence
            # Allow up to 2 words in between each phrase word
            current_pos = i
            for phrase_word in phrase_words:
                found = False
                # Look ahead up to 3 positions for the next word
                for j in range(current_pos, min(current_pos + 3, len(words))):
                    if words[j] == phrase_word:
                        matched_positions.append(j)
                        current_pos = j + 1
                        found = True
                        break
                
                if not found:
                    matched = False
                    break
            
            if matched:
                # Calculate confidence based on how compact the match is
                span = matched_positions[-1] - matched_positions[0] + 1
                compactness = phrase_len / span
                confidence_boost = 0.3 + (0.2 * compactness)  # 0.3 to 0.5 based on compactness
                return True, confidence_boost
        
        return False, 0.0
    
    def detect_bullying(self, text: str, confidence_threshold: float = 0.3) -> Dict:
        """
        Main detection method with multi-language support
        """
        self.stats['total_detections'] += 1
        
        if not text or not text.strip():
            return {
                'is_bullying': False,
                'confidence': 0.0,
                'severity': 'none',
                'detected_languages': [],
                'detected_items': [],
                'detection_method': 'empty_text'
            }
        
        # Tokenize text
        words = self._tokenize_text(text)
        
        # Detect languages
        detected_languages = self._detect_language(text)
        for lang in detected_languages:
            self.stats['languages_detected'][lang] += 1
        
        # Detection results
        all_detections = []
        total_confidence = 0.0
        max_severity = 'none'
        severity_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        
        # Check each detected language
        for language in detected_languages:
            if language not in self.language_data:
                continue
            
            lang_data = self.language_data[language]
            language_detections = []
            
            # Check single words
            for word in words:
                if word in lang_data['single_words']:
                    severity = lang_data['severity_map'].get(word, 'medium')
                    confidence_boost = 0.3 if severity == 'low' else 0.4 if severity == 'medium' else 0.5
                    
                    language_detections.append({
                        'type': 'single_word',
                        'value': word,
                        'language': language,
                        'severity': severity,
                        'confidence': confidence_boost
                    })
                    total_confidence += confidence_boost
                    self.stats['word_matches'] += 1
            
            # Check multi-word phrases
            for phrase_data in lang_data['phrases']:
                found, confidence_boost = self._check_multi_word_phrase(words, phrase_data)
                if found:
                    severity = phrase_data['severity']
                    # Higher confidence for phrase matches
                    confidence_boost = confidence_boost * 1.5
                    
                    language_detections.append({
                        'type': 'phrase',
                        'value': phrase_data['phrase'],
                        'language': language,
                        'severity': severity,
                        'confidence': confidence_boost
                    })
                    total_confidence += confidence_boost
                    self.stats['phrase_matches'] += 1
            
            # Check regex patterns
            text_lower = text.lower()
            for pattern in lang_data['patterns']:
                matches = pattern.findall(text_lower)
                if matches:
                    for match in matches:
                        language_detections.append({
                            'type': 'pattern',
                            'value': match,
                            'language': language,
                            'severity': 'medium',
                            'confidence': 0.4
                        })
                        total_confidence += 0.4
            
            all_detections.extend(language_detections)
        
        # Calculate final results
        if all_detections:
            # Normalize confidence (cap at 1.0)
            total_confidence = min(total_confidence, 1.0)
            
            # Determine max severity
            for detection in all_detections:
                detection_severity = detection.get('severity', 'medium')
                if severity_levels.get(detection_severity, 0) > severity_levels.get(max_severity, 0):
                    max_severity = detection_severity
            
            # Boost confidence for multiple detections
            if len(all_detections) > 2:
                total_confidence = min(total_confidence * 1.2, 1.0)
        
        is_bullying = total_confidence >= confidence_threshold
        
        if is_bullying:
            self.stats['bullying_detected'] += 1
        
        return {
            'is_bullying': is_bullying,
            'confidence': total_confidence,
            'severity': max_severity if is_bullying else 'none',
            'detected_languages': detected_languages,
            'detected_items': all_detections,
            'detection_method': 'multilingual',
            'word_count': len(words),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def add_words(self, words: List[str], language: str = 'english', word_type: str = 'single') -> int:
        """Add new words or phrases to the database"""
        if language not in self.bullying_data.get('languages', {}):
            self.bullying_data.setdefault('languages', {})[language] = {
                'single_words': [],
                'multi_word_phrases': [],
                'patterns': [],
                'severity_map': {}
            }
        
        added_count = 0
        target_list = 'single_words' if word_type == 'single' else 'multi_word_phrases'
        current_items = set(self.bullying_data['languages'][language].get(target_list, []))
        
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in current_items:
                self.bullying_data['languages'][language][target_list].append(word_clean)
                current_items.add(word_clean)
                added_count += 1
        
        # Reprocess data
        if added_count > 0:
            self._preprocess_multilingual_data()
            self._save_database()
        
        return added_count
    
    def _save_database(self):
        """Save the updated database"""
        try:
            self.bullying_data['metadata']['last_updated'] = datetime.utcnow().isoformat()
            
            os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
            
            with open(self.data_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.bullying_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Database saved successfully")
        except Exception as e:
            logger.error(f"Error saving database: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        self.stats['last_updated'] = datetime.utcnow().isoformat()
        return self.stats.copy()
