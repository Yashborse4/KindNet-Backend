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
        """Load multilingual bullying database.

        We always start from the built-in defaults (which contain a sensible
        set of English/Hindi/Marathi words & phrases) and then merge any
        locally stored JSON data on top. This ensures that upgrading the
        detector does not silently lose important default phrases like
        "kill yourself" or "nobody likes you" even if the on-disk JSON was
        saved in an earlier, minimal "languages" format.
        """
        try:
            # Base data shipped with the code
            base_data = self._get_default_multilingual_data()

            if not os.path.exists(self.data_file_path):
                logger.warning(f"Database file not found: {self.data_file_path}")
                return base_data

            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # If this is an old "categories"-based file, convert it first
            if 'languages' not in data:
                data = self._convert_to_multilingual_format(data)

            # Merge loaded data on top of the defaults
            merged = self._merge_multilingual_data(base_data, data)
            return merged

        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            return self._get_default_multilingual_data()
    
    def _get_default_multilingual_data(self) -> Dict:
        """Default multilingual data structure.

        This is used as the baseline and then merged with any on-disk
        configuration so that important default phrases for English/Hindi/
        Marathi are always available.
        """
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
                    # Romanised Hindi insults/offensive terms
                    'single_words': ['chutiya', 'gandu', 'harami', 'kutte', 'nalayak', 'gaand'],
                    # Common Hindi abusive phrases (romanised)
                    'multi_word_phrases': ['teri maa ki', 'ullu ka patha', 'gaand maaru', 'teri maa ka'],
                    # Regex patterns for both romanised and Devanagari variants
                    'patterns': [
                        # Romanised insults
                        r'\bteri\s+maa\s+ki\b',
                        r'\bgaand\s+maa?r',
                        # Devanagari equivalents used in tests / local content
                        r'तेरी\s*माँ\s*की',
                        r'चुतिया',
                    ],
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
        """Convert legacy format to multilingual format.

        This function takes the older "bullying_categories" based JSON and
        populates the multilingual structure used by this detector.
        """
        multilingual_data = self._get_default_multilingual_data()

        # Extract all words and phrases from legacy format
        all_words: List[str] = []
        all_phrases: List[str] = []

        # From top-level bullying_words (if present)
        if 'bullying_words' in data:
            all_words.extend(data['bullying_words'])

        # From top-level bullying_phrases (if present)
        if 'bullying_phrases' in data:
            all_phrases.extend(data['bullying_phrases'])

        # From category-based structure
        if 'bullying_categories' in data:
            for category, cat_data in data['bullying_categories'].items():
                # Single keywords
                if 'keywords' in cat_data:
                    for keyword in cat_data['keywords']:
                        keyword = keyword.strip()
                        if not keyword:
                            continue
                        if ' ' in keyword:
                            all_phrases.append(keyword)
                        else:
                            all_words.append(keyword)

                # Multi‑word phrases inside categories (previously ignored)
                # This is critical for phrases like "kill yourself", "go die",
                # "nobody likes you", "ullu ka patha", "aai ghalya", etc.
                if 'multi_word_phrases' in cat_data:
                    for phrase in cat_data['multi_word_phrases']:
                        phrase = phrase.strip()
                        if phrase:
                            all_phrases.append(phrase)

        # Categorize words by language
        hindi_words: List[str] = []
        hindi_phrases: List[str] = []
        english_words: List[str] = []
        english_phrases: List[str] = []

        # Heuristic list of romanised Hindi markers used to classify entries
        hindi_patterns = [
            'chutiya', 'bhosdike', 'gandu', 'lund', 'gaand', 'randi', 'harami',
            'kutte', 'suar', 'nalayak', 'bevakuf', 'ullu', 'kaminey', 'madarchod',
            'bhenchod', 'chod', 'teri', 'maa', 'behen', 'baap', 'bewakoof'
        ]

        # Certain words are highly context‑dependent and should not be treated as
        # bullying on their own (e.g. "nobody" as in "nobody knows"). These are
        # handled via phrases/patterns instead.
        context_sensitive_words: Set[str] = {
            'nobody',
        }

        # Categorize single words
        for word in all_words:
            word_lower = word.lower().strip()
            if not word_lower:
                continue

            # Skip context‑sensitive words here; they will still be covered by
            # phrase/pattern based detection where appropriate.
            if word_lower in context_sensitive_words:
                continue

            if any(hindi_word in word_lower for hindi_word in hindi_patterns):
                hindi_words.append(word_lower)
            else:
                english_words.append(word_lower)

        # Categorize phrases
        for phrase in all_phrases:
            phrase_lower = phrase.lower().strip()
            if not phrase_lower:
                continue

            if any(hindi_word in phrase_lower for hindi_word in hindi_patterns):
                hindi_phrases.append(phrase_lower)
            else:
                english_phrases.append(phrase_lower)

        # Update multilingual data (preserve Marathi and other languages)
        multilingual_data['languages']['english']['single_words'] = list(set(english_words))
        multilingual_data['languages']['english']['multi_word_phrases'] = list(set(english_phrases))
        multilingual_data['languages']['hindi']['single_words'] = list(set(hindi_words))
        multilingual_data['languages']['hindi']['multi_word_phrases'] = list(set(hindi_phrases))
        
        return multilingual_data

    def _merge_multilingual_data(self, base: Dict, override: Dict) -> Dict:
        """Merge two multilingual data dictionaries.

        - List fields (single_words, multi_word_phrases, patterns) are merged
          with de-duplication.
        - severity_map is updated with override taking precedence.
        - New languages in override are added as-is.
        """
        result = base

        # Merge languages
        for lang, lang_data in override.get('languages', {}).items():
            base_lang = result.setdefault('languages', {}).setdefault(
                lang,
                {
                    'single_words': [],
                    'multi_word_phrases': [],
                    'patterns': [],
                    'severity_map': {},
                },
            )

            # Merge list-like fields
            for key in ('single_words', 'multi_word_phrases', 'patterns'):
                base_list = list(base_lang.get(key, []))
                override_list = list(lang_data.get(key, []))
                # Preserve order while de-duplicating
                merged_list = list(dict.fromkeys(base_list + override_list))
                base_lang[key] = merged_list

            # Merge severity maps (override wins)
            base_severity = dict(base_lang.get('severity_map', {}))
            base_severity.update(lang_data.get('severity_map', {}))
            base_lang['severity_map'] = base_severity

        # Merge metadata
        if 'metadata' in override:
            result.setdefault('metadata', {}).update(override['metadata'])

        return result
    
    def _preprocess_multilingual_data(self):
        """Preprocess data for efficient multi-language lookup"""
        self.language_data = {}

        # Words that are highly context-dependent and should not be treated as
        # bullying on their own for specific languages. They are still handled
        # via phrases/patterns where context is clearer.
        context_sensitive_single_exclude = {
            'english': {
                # Treat these via phrases/patterns only so that technical
                # sentences like "kill the process" or "die casting" are not
                # automatically flagged.
                'nobody',
                'die',
                'kill',
            },
        }
        
        for language, lang_data in self.bullying_data.get('languages', {}).items():
            # Create sets for O(1) lookup
            excluded = context_sensitive_single_exclude.get(language, set())
            single_words_set = set(
                word.lower()
                for word in lang_data.get('single_words', [])
                if word and word.lower() not in excluded
            )
            
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
        """Detect which languages are present in the text.

        The original implementation relied on a small, hard‑coded list of
        romanised words. That caused issues for phrases added at runtime (e.g.
        "tera baap ka") and for some Marathi/Hindi cases. Here we use the
        actual loaded database for language hints, while still supporting
        Devanagari detection.
        """
        detected_languages: Set[str] = set()

        if not text:
            return []

        text_lower = text.lower()
        # Re‑use the same tokenisation we use for detection so behaviour is
        # consistent across languages, including Hindi/Marathi.
        words = self._tokenize_text(text_lower)

        # Hindi/Devanagari script detection – this also covers Marathi written
        # in Devanagari, but we map it to Hindi for now since the database is
        # primarily Hindi‑oriented.
        if re.search(r'[\u0900-\u097F]', text):
            detected_languages.add('hindi')

        # Use the multilingual database itself as a source of hints: any word,
        # phrase or pattern that matches marks that language as present.
        for language, lang_data in self.language_data.items():
            # Single‑word hints
            if any(word in lang_data['single_words'] for word in words):
                detected_languages.add(language)
                continue

            # Phrase hints – check if any phrase appears as a substring
            for phrase_info in lang_data['phrases']:
                phrase = phrase_info.get('phrase')
                if phrase and phrase in text_lower:
                    detected_languages.add(language)
                    break

            if language in detected_languages:
                continue

            # Regex pattern hints
            for pattern in lang_data['patterns']:
                if pattern.search(text_lower):
                    detected_languages.add(language)
                    break

        # Always include English as a fallback for mixed/unknown content to keep
        # backwards‑compatible behaviour.
        detected_languages.add('english')

        return list(detected_languages)
    
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

            # Special handling for common Hindi Devanagari insults that may not
            # appear in the Romanised word lists but are frequent in local
            # content (e.g. social media posts in Hindi script).
            if language == 'hindi':
                devanagari_map = {
                    'चुतिया': 'medium',
                    'चूतिया': 'medium',  # common spelling variant
                    'तेरी माँ की': 'high',
                    'तेरी मां की': 'high',
                }
                for phrase, severity in devanagari_map.items():
                    if phrase in text:
                        confidence_boost = 0.4 if severity == 'medium' else 0.5
                        language_detections.append({
                            'type': 'devanagari_phrase',
                            'value': phrase,
                            'language': language,
                            'severity': severity,
                            'confidence': confidence_boost,
                        })
                        total_confidence += confidence_boost
                        self.stats['phrase_matches'] += 1
            
            # Check single words
            for word in words:
                if word in lang_data['single_words']:
                    severity = lang_data['severity_map'].get(word, 'medium')
                    confidence_boost = 0.3 if severity == 'low' else 0.4 if severity == 'medium' else 0.8
                    
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
