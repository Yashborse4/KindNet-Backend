import json
import re
import os
import logging
import threading
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import openai
from openai import OpenAI
from collections import defaultdict, Counter, deque, OrderedDict
import string
import nltk
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU Cache implementation with size limits."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache, updating access order."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: str) -> None:
        """Put value in cache, evicting LRU if necessary."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
            
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0.0
            }


class AhoCorasickAutomaton:
    """Efficient Aho-Corasick automaton for multi-pattern matching.
    
    Provides O(n + m + z) time complexity where:
    - n is the length of the text
    - m is the total length of all patterns
    - z is the number of matches found
    
    This replaces the O(n²) phrase sequence checking algorithm.
    """
    
    def __init__(self):
        self.goto = {}  # State transition function
        self.failure = {}  # Failure function for mismatches
        self.output = {}  # Output function for pattern matches
        self.patterns = []  # Store original patterns with metadata
        self.state_count = 0
        self.root = 0
        self._built = False
        
    def add_pattern(self, pattern: str, category: str, confidence_boost: float = 0.3, 
                   severity: str = 'medium', max_gap: int = 2) -> None:
        """Add a pattern to the automaton.
        
        Args:
            pattern: The pattern to match (can be multi-word)
            category: Category this pattern belongs to
            confidence_boost: Confidence boost for matches
            severity: Severity level of this pattern
            max_gap: Maximum allowed words between pattern components
        """
        if self._built:
            raise RuntimeError("Cannot add patterns after automaton is built")
            
        pattern_words = pattern.lower().split()
        if not pattern_words:
            return
            
        self.patterns.append({
            'pattern': pattern,
            'words': pattern_words, 
            'category': category,
            'confidence_boost': confidence_boost,
            'severity': severity,
            'max_gap': max_gap,
            'pattern_id': len(self.patterns)
        })
    
    def build(self) -> None:
        """Build the Aho-Corasick automaton from added patterns."""
        if self._built:
            return
            
        # Initialize root state
        self.goto[self.root] = {}
        current_state = 0
        
        # Build goto function (trie construction)
        for pattern_info in self.patterns:
            words = pattern_info['words']
            state = self.root
            
            for word in words:
                if word not in self.goto[state]:
                    current_state += 1
                    self.goto[state][word] = current_state
                    self.goto[current_state] = {}
                
                state = self.goto[state][word]
            
            # Mark this state as accepting this pattern
            if state not in self.output:
                self.output[state] = []
            self.output[state].append(pattern_info)
        
        self.state_count = current_state + 1
        
        # Build failure function using BFS
        self._build_failure_function()
        self._built = True
    
    def _build_failure_function(self) -> None:
        """Build failure function for pattern matching."""
        queue = deque()
        
        # Initialize failure function for depth-1 states
        for word in self.goto[self.root]:
            state = self.goto[self.root][word]
            self.failure[state] = self.root
            queue.append(state)
        
        # Build failure function for deeper states
        while queue:
            r = queue.popleft()
            
            for word in self.goto[r]:
                s = self.goto[r][word]
                queue.append(s)
                
                # Find failure state
                state = self.failure[r]
                while state != self.root and word not in self.goto[state]:
                    state = self.failure[state]
                
                if word in self.goto[state] and self.goto[state][word] != s:
                    self.failure[s] = self.goto[state][word]
                else:
                    self.failure[s] = self.root
                
                # Add output from failure state
                failure_state = self.failure[s]
                if failure_state in self.output:
                    if s not in self.output:
                        self.output[s] = []
                    self.output[s].extend(self.output[failure_state])
    
    def search(self, text: str) -> List[Dict]:
        """Search for all patterns in the text with gap tolerance.
        
        Returns list of matches with position, pattern info, and gap analysis.
        """
        if not self._built:
            self.build()
        
        if not text.strip():
            return []
            
        words = text.lower().split()
        if not words:
            return []
            
        matches = []
        
        # For each starting position, try to match patterns with gap tolerance
        for start_pos in range(len(words)):
            matches.extend(self._search_from_position(words, start_pos))
        
        # Deduplicate matches (same pattern at overlapping positions)
        unique_matches = self._deduplicate_matches(matches)
        
        return unique_matches
    
    def _search_from_position(self, words: List[str], start_pos: int) -> List[Dict]:
        """Search for patterns starting from a specific position with gap tolerance."""
        matches = []
        
        for pattern_info in self.patterns:
            match = self._match_pattern_with_gaps(words, start_pos, pattern_info)
            if match:
                matches.append(match)
        
        return matches
    
    def _match_pattern_with_gaps(self, words: List[str], start_pos: int, 
                                pattern_info: Dict) -> Optional[Dict]:
        """Advanced pattern matching with adaptive gap limits and importance weighting."""
        pattern_words = pattern_info['words']
        base_max_gap = pattern_info['max_gap']
        
        if start_pos >= len(words):
            return None
        
        # Adaptive gap limits based on pattern characteristics
        adaptive_gap_limit = self._calculate_adaptive_gap_limit(
            pattern_words, base_max_gap, len(words)
        )
        
        matched_positions = []
        current_pos = start_pos
        gap_penalties = []
        word_importance_scores = []
        
        # Try to match each word in the pattern with intelligent gap handling
        for word_idx, pattern_word in enumerate(pattern_words):
            found_pos = None
            word_gap = 0
            
            # Calculate importance weight for this word
            word_importance = self._calculate_word_importance(
                pattern_word, word_idx, pattern_words
            )
            word_importance_scores.append(word_importance)
            
            # Adaptive search window based on word importance and position
            search_window = self._calculate_search_window(
                adaptive_gap_limit, word_importance, word_idx, len(pattern_words)
            )
            
            search_end = min(current_pos + search_window + 1, len(words))
            
            # Look for pattern word with preference for closer matches
            for pos in range(current_pos, search_end):
                if words[pos] == pattern_word:
                    found_pos = pos
                    word_gap = pos - current_pos
                    break
            
            if found_pos is None:
                return None  # Pattern word not found within adaptive gap limit
            
            matched_positions.append(found_pos)
            gap_penalties.append(self._calculate_gap_penalty(word_gap, word_importance))
            current_pos = found_pos + 1
        
        # Advanced match quality calculation
        match_quality = self._calculate_advanced_match_quality(
            matched_positions, pattern_words, gap_penalties, 
            word_importance_scores, len(words)
        )
        
        return {
            'pattern': pattern_info['pattern'],
            'category': pattern_info['category'],
            'confidence_boost': pattern_info['confidence_boost'] * match_quality['strength'],
            'severity': pattern_info['severity'],
            'start_pos': matched_positions[0],
            'end_pos': matched_positions[-1],
            'span': match_quality['span'],
            'gaps': match_quality['total_gaps'],
            'match_strength': match_quality['strength'],
            'matched_positions': matched_positions,
            'quality_metrics': match_quality,
            'word_importance': word_importance_scores
        }
    
    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate and overlapping matches, keeping the strongest."""
        if not matches:
            return []
        
        # Group matches by pattern
        pattern_groups = defaultdict(list)
        for match in matches:
            pattern_groups[match['pattern']].append(match)
        
        unique_matches = []
        
        # For each pattern, keep only the best non-overlapping matches
        for pattern, pattern_matches in pattern_groups.items():
            # Sort by match strength (descending) then by position
            pattern_matches.sort(key=lambda x: (-x['match_strength'], x['start_pos']))
            
            selected = []
            for match in pattern_matches:
                # Check if this match overlaps significantly with any selected match
                overlaps = False
                for selected_match in selected:
                    if self._matches_overlap(match, selected_match):
                        overlaps = True
                        break
                
                if not overlaps:
                    selected.append(match)
            
            unique_matches.extend(selected)
        
        return unique_matches
    
    def _matches_overlap(self, match1: Dict, match2: Dict, 
                        overlap_threshold: float = 0.5) -> bool:
        """Check if two matches overlap significantly."""
        start1, end1 = match1['start_pos'], match1['end_pos']
        start2, end2 = match2['start_pos'], match2['end_pos']
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_length = max(0, overlap_end - overlap_start + 1)
        
        # Calculate overlap ratio relative to smaller match
        min_length = min(end1 - start1 + 1, end2 - start2 + 1)
        overlap_ratio = overlap_length / min_length if min_length > 0 else 0
        
        return overlap_ratio > overlap_threshold
    
    def _calculate_adaptive_gap_limit(self, pattern_words: List[str], 
                                     base_max_gap: int, text_length: int) -> int:
        """Calculate adaptive gap limit based on pattern characteristics.
        
        Factors considered:
        - Pattern length (longer patterns allow more gaps)
        - Text length (shorter texts are stricter)
        - Word importance (patterns with important words get more flexibility)
        """
        pattern_length = len(pattern_words)
        
        # Base adaptive multiplier based on pattern length
        if pattern_length <= 2:
            length_multiplier = 1.0  # Short patterns stay strict
        elif pattern_length <= 4:
            length_multiplier = 1.2  # Medium patterns get slight flexibility
        else:
            length_multiplier = 1.5  # Long patterns get more flexibility
        
        # Text length factor - shorter texts are stricter
        if text_length < 10:
            text_multiplier = 0.8
        elif text_length < 20:
            text_multiplier = 1.0
        else:
            text_multiplier = 1.2
        
        # Pattern importance factor
        importance_multiplier = 1.0
        high_importance_words = {'kill', 'die', 'hate', 'stupid', 'ugly', 'worthless'}
        
        if any(word in high_importance_words for word in pattern_words):
            importance_multiplier = 1.3  # More flexible for important patterns
        
        # Calculate final adaptive limit
        adaptive_limit = int(base_max_gap * length_multiplier * text_multiplier * importance_multiplier)
        
        # Ensure reasonable bounds
        return max(1, min(adaptive_limit, 5))
    
    def _calculate_word_importance(self, word: str, position: int, 
                                  pattern_words: List[str]) -> float:
        """Calculate the importance weight of a word in the pattern.
        
        Factors considered:
        - Semantic importance (threatening words, core insults)
        - Position in pattern (first/last words are more important)
        - Word uniqueness (less common words are more important)
        - Word length (very short words are less important)
        """
        importance = 0.5  # Base importance
        
        # Semantic importance based on word content
        high_importance_words = {
            'kill', 'die', 'suicide', 'hurt', 'harm', 'destroy', 'end',
            'stupid', 'idiot', 'ugly', 'worthless', 'pathetic', 'hate',
            'loser', 'freak', 'nobody', 'alone'
        }
        
        medium_importance_words = {
            'you', 'your', 'yourself', 'are', 'should', 'must',
            'go', 'get', 'make', 'feel'
        }
        
        if word in high_importance_words:
            importance += 0.4  # High semantic importance
        elif word in medium_importance_words:
            importance += 0.2  # Medium semantic importance
        
        # Position importance (first and last words are anchors)
        pattern_length = len(pattern_words)
        if position == 0 or position == pattern_length - 1:
            importance += 0.2  # First or last word bonus
        elif position == 1 or position == pattern_length - 2:
            importance += 0.1  # Second/second-to-last bonus
        
        # Word length factor (very short words are less reliable)
        if len(word) <= 2:
            importance -= 0.2  # Penalty for very short words
        elif len(word) >= 6:
            importance += 0.1  # Bonus for longer, more specific words
        
        # Ensure reasonable bounds
        return max(0.1, min(importance, 1.0))
    
    def _calculate_search_window(self, adaptive_gap_limit: int, word_importance: float,
                               word_position: int, pattern_length: int) -> int:
        """Calculate the search window size for finding the next pattern word.
        
        More important words get wider search windows.
        Words at the beginning/end of patterns get wider windows.
        """
        base_window = adaptive_gap_limit
        
        # Importance multiplier
        importance_multiplier = 0.5 + word_importance  # Range: 0.6 to 1.5
        
        # Position multiplier (edges get more flexibility)
        if word_position == 0 or word_position == pattern_length - 1:
            position_multiplier = 1.3
        else:
            position_multiplier = 1.0
        
        # Calculate final window
        search_window = int(base_window * importance_multiplier * position_multiplier)
        
        # Ensure reasonable bounds
        return max(1, min(search_window, 8))
    
    def _calculate_gap_penalty(self, gap_size: int, word_importance: float) -> float:
        """Calculate penalty for gaps between pattern words.
        
        Less important words get higher penalties for gaps.
        Larger gaps always get higher penalties.
        """
        if gap_size == 0:
            return 0.0  # No gap, no penalty
        
        # Base penalty increases with gap size
        base_penalty = min(gap_size * 0.15, 0.6)  # Max 60% penalty
        
        # Adjust penalty based on word importance
        importance_adjustment = (1.0 - word_importance) * 0.3  # Up to 30% additional penalty
        
        total_penalty = base_penalty + importance_adjustment
        
        return min(total_penalty, 0.8)  # Cap at 80% penalty
    
    def _calculate_advanced_match_quality(self, matched_positions: List[int],
                                         pattern_words: List[str], 
                                         gap_penalties: List[float],
                                         word_importance_scores: List[float],
                                         text_length: int) -> Dict:
        """Calculate advanced match quality metrics with comprehensive scoring."""
        if not matched_positions:
            return {
                'strength': 0.0,
                'span': 0,
                'total_gaps': 0,
                'quality_score': 0.0,
                'compactness': 0.0,
                'importance_score': 0.0
            }
        
        # Basic metrics
        span = matched_positions[-1] - matched_positions[0] + 1
        total_gaps = sum(matched_positions[i+1] - matched_positions[i] - 1 
                        for i in range(len(matched_positions) - 1))
        
        # Compactness score (how tightly packed the match is)
        expected_span = len(pattern_words)
        compactness = expected_span / span if span > 0 else 0
        
        # Importance-weighted score
        avg_importance = sum(word_importance_scores) / len(word_importance_scores)
        
        # Gap penalty score (average of all gap penalties)
        avg_gap_penalty = sum(gap_penalties) / len(gap_penalties) if gap_penalties else 0
        
        # Position quality (matches closer to text start/end might be more reliable)
        text_start_distance = matched_positions[0] / max(text_length, 1)
        text_end_distance = (text_length - matched_positions[-1]) / max(text_length, 1)
        position_quality = 1.0 - min(text_start_distance, text_end_distance) * 0.1  # Slight penalty for edge positions
        
        # Calculate overall quality score
        quality_components = {
            'compactness': compactness * 0.3,
            'importance': avg_importance * 0.25,
            'gap_penalty': (1.0 - avg_gap_penalty) * 0.25,  # Invert penalty to positive score
            'position': position_quality * 0.2
        }
        
        quality_score = sum(quality_components.values())
        
        # Calculate final match strength
        # Combine quality score with pattern-specific adjustments
        pattern_length_bonus = min(len(pattern_words) * 0.05, 0.15)  # Bonus for longer patterns
        
        match_strength = quality_score + pattern_length_bonus
        match_strength = max(0.1, min(match_strength, 1.0))  # Ensure bounds
        
        return {
            'strength': match_strength,
            'span': span,
            'total_gaps': total_gaps,
            'quality_score': quality_score,
            'compactness': compactness,
            'importance_score': avg_importance,
            'gap_penalty_score': avg_gap_penalty,
            'position_quality': position_quality,
            'quality_components': quality_components
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the automaton."""
        return {
            'pattern_count': len(self.patterns),
            'state_count': self.state_count,
            'built': self._built,
            'categories': list(set(p['category'] for p in self.patterns))
        }

class IntelligentBullyingDetector:
    """
    Advanced AI-powered cyberbullying detection system with:
    1. Context-aware analysis
    2. Sentiment analysis
    3. Intent classification
    4. Multi-model ensemble
    5. Adaptive learning
    """
    
    def __init__(self, data_file_path: str = None, openai_api_key: str = None):
        """Initialize the intelligent detector"""
        self.data_file_path = data_file_path or os.path.join('data', 'bullying_words.json')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Thread safety
        self._stats_lock = threading.RLock()
        
        # Enhanced statistics tracking with thread-safe access
        self.stats = {
            'total_detections': 0,
            'bullying_detected': 0,
            'false_positives': 0,
            'confidence_distribution': defaultdict(int),
            'category_counts': defaultdict(int),
            'severity_distribution': defaultdict(int),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Load enhanced database
        self.bullying_data = self._load_enhanced_database()
        
        # Initialize AI components
        self._init_openai_client()
        self._init_nlp_components()
        self._preprocess_enhanced_data()
        
        logger.info("IntelligentBullyingDetector initialized successfully")
    
    def _load_enhanced_database(self) -> Dict:
        """Load enhanced bullying database with reliable format detection"""
        try:
            if not os.path.exists(self.data_file_path):
                # Ensure directory exists first to prevent FileNotFoundError
                os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
                logger.info("Database file not found, creating default enhanced format")
                return self._get_enhanced_default_data()
            
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine format and convert if necessary
            format_type = self._detect_database_format(data)
            logger.info(f"Detected database format: {format_type}")
            
            if format_type == 'enhanced':
                # Already in enhanced format, validate and fill missing fields
                data = self._validate_enhanced_format(data)
                return data
            elif format_type in ['legacy_words', 'legacy_languages']:
                # Convert legacy format to enhanced
                logger.info(f"Converting {format_type} to enhanced format")
                converted_data = self._convert_legacy_data(data)
                # Save converted data to prevent repeated conversions
                self._save_database_safely(converted_data)
                return converted_data
            else:
                logger.warning(f"Unknown database format, using default")
                return self._get_enhanced_default_data()
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in database file: {str(e)}")
            return self._get_enhanced_default_data()
        except Exception as e:
            logger.error(f"Error loading database: {str(e)}")
            return self._get_enhanced_default_data()
    
    def _get_enhanced_default_data(self) -> Dict:
        """Enhanced default data with categorized detection"""
        # Try to load from existing bullying_words.json structure
        try:
            if os.path.exists(self.data_file_path):
                with open(self.data_file_path, 'r', encoding='utf-8') as f:
                    legacy_data = json.load(f)
                    if 'bullying_words' in legacy_data:
                        # Convert legacy format to new format
                        logger.info(f"Converting legacy data format with {len(legacy_data.get('bullying_words', []))} words")
                        return self._convert_legacy_data(legacy_data)
        except Exception as e:
            logger.warning(f"Could not load legacy data: {str(e)}")
        
        return {
            'bullying_categories': {
                'direct_insults': {
                    'keywords': ['stupid', 'idiot', 'loser', 'ugly', 'fat', 'worthless', 'pathetic'],
                    'patterns': [r'\byou\s+are\s+(so\s+)?(stupid|ugly|worthless)', r'\bgo\s+die\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.3
                },
                'threats': {
                    'keywords': ['kill', 'hurt', 'harm', 'destroy', 'end you'],
                    'patterns': [r'\bkill\s+yourself\b', r'\bi\s+will\s+hurt\s+you\b'],
                    'severity': 'high',
                    'confidence_boost': 0.5
                },
                'social_exclusion': {
                    'keywords': ['nobody likes you', 'friendless', 'alone', 'outcast'],
                    'patterns': [r'\bnobody\s+(likes|wants)\s+you\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.4
                },
                'harassment': {
                    'keywords': ['freak', 'weirdo', 'creep', 'stalker'],
                    'patterns': [r'\byou\s+(creep|freak)\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.35
                }
            },
            'severity_patterns': {
                'high': [r'\b(kill\s+yourself|end\s+your\s+life|go\s+die)\b'],
                'medium': [r'\b(stupid|ugly|worthless|pathetic)\b'],
                'low': [r'\b(weird|annoying|dumb)\b']
            },
            'context_indicators': {
                'aggressive_tone': ['!!!', 'CAPS LOCK', 'excessive punctuation'],
                'targeting': ['you', 'your', 'yourself'],
                'intent_markers': ['should', 'need to', 'must', 'have to']
            },
            'emotional_markers': {
                'anger': ['hate', 'angry', 'furious', 'rage'],
                'disgust': ['disgusting', 'gross', 'sick', 'revolting'],
                'contempt': ['pathetic', 'worthless', 'useless', 'waste']
            }
        }
    
    def _init_nlp_components(self):
        """Initialize NLP components for advanced analysis"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TF-IDF vectorizer for similarity analysis
            self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            
            logger.info("NLP components initialized")
        except Exception as e:
            logger.warning(f"NLP initialization failed: {str(e)}")
            self.sentiment_analyzer = None
            self.tfidf = None
    
    def _init_openai_client(self):
        """Initialize OpenAI client with enhanced prompts"""
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"OpenAI initialization error: {str(e)}")
                self.openai_client = None
        else:
            self.openai_client = None
    
    def _preprocess_enhanced_data(self):
        """Enhanced data preprocessing with categorization and advanced caching"""
        self.category_patterns = {}
        self.severity_regex = {}
        
        # Initialize pattern caching structures
        self._pattern_cache = {}
        self._normalized_text_cache = LRUCache(max_size=1000)  # Use LRU cache for text normalization
        self._keyword_lookup = {}
        self._phrase_index = {}
        
        # Initialize optimized phrase detection automaton
        self.phrase_automaton = AhoCorasickAutomaton()
        
        # Compile category patterns with caching
        for category, data in self.bullying_data.get('bullying_categories', {}).items():
            # Process keywords - separate single words and multi-word phrases
            single_words = set()
            multi_word_phrases = []
            
            for keyword in data.get('keywords', []):
                keyword_lower = keyword.lower().strip()
                # Check if it's a multi-word phrase
                if ' ' in keyword_lower:
                    multi_word_phrases.append(keyword_lower)
                    # Add phrase to automaton for efficient detection
                    self.phrase_automaton.add_pattern(
                        keyword_lower, 
                        category,
                        data.get('confidence_boost', 0.3),
                        data.get('severity', 'medium'),
                        max_gap=2  # Allow up to 2 words between phrase components
                    )
                    # Also add individual words from the phrase to catch partial matches
                    words_in_phrase = keyword_lower.split()
                    single_words.update(words_in_phrase)
                    
                    # Add to phrase index for quick lookup
                    phrase_key = self._generate_phrase_key(words_in_phrase)
                    if phrase_key not in self._phrase_index:
                        self._phrase_index[phrase_key] = []
                    self._phrase_index[phrase_key].append({
                        'phrase': keyword_lower,
                        'category': category,
                        'confidence_boost': data.get('confidence_boost', 0.3),
                        'severity': data.get('severity', 'medium')
                    })
                else:
                    single_words.add(keyword_lower)
                    # Add to keyword lookup for O(1) access
                    self._keyword_lookup[keyword_lower] = {
                        'category': category,
                        'confidence_boost': data.get('confidence_boost', 0.3),
                        'severity': data.get('severity', 'medium')
                    }
            
            # Add explicit multi-word phrases from the data
            for phrase in data.get('multi_word_phrases', []):
                phrase_lower = phrase.lower().strip()
                multi_word_phrases.append(phrase_lower)
                self.phrase_automaton.add_pattern(
                    phrase_lower,
                    category,
                    data.get('confidence_boost', 0.3) * 1.5,  # Boost for explicit phrases
                    data.get('severity', 'medium'),
                    max_gap=2
                )
                
                # Add to phrase index
                words_in_phrase = phrase_lower.split()
                phrase_key = self._generate_phrase_key(words_in_phrase)
                if phrase_key not in self._phrase_index:
                    self._phrase_index[phrase_key] = []
                self._phrase_index[phrase_key].append({
                    'phrase': phrase_lower,
                    'category': category,
                    'confidence_boost': data.get('confidence_boost', 0.3) * 1.5,
                    'severity': data.get('severity', 'medium')
                })
            
            # Cache compiled patterns
            compiled_patterns = []
            for pattern in data.get('patterns', []):
                pattern_key = f"regex_{category}_{hash(pattern)}"
                if pattern_key not in self._pattern_cache:
                    self._pattern_cache[pattern_key] = re.compile(pattern, re.IGNORECASE)
                compiled_patterns.append(self._pattern_cache[pattern_key])
            
            self.category_patterns[category] = {
                'keywords': single_words,
                'multi_word_phrases': multi_word_phrases,
                'patterns': compiled_patterns,
                'severity': data.get('severity', 'medium'),
                'confidence_boost': data.get('confidence_boost', 0.3)
            }
        
        # Build the automaton for efficient searching
        self.phrase_automaton.build()
        
        # Cache severity patterns
        for severity, patterns in self.bullying_data.get('severity_patterns', {}).items():
            cached_patterns = []
            for pattern in patterns:
                pattern_key = f"severity_{severity}_{hash(pattern)}"
                if pattern_key not in self._pattern_cache:
                    self._pattern_cache[pattern_key] = re.compile(pattern, re.IGNORECASE)
                cached_patterns.append(self._pattern_cache[pattern_key])
            self.severity_regex[severity] = cached_patterns
        
        # Pre-compile common normalization patterns
        self._precompile_normalization_patterns()
        
        # Log preprocessing statistics
        stats = self.phrase_automaton.get_stats()
        logger.info(f"Enhanced preprocessing complete:")
        logger.info(f"  - Phrase automaton: {stats['pattern_count']} patterns, {stats['state_count']} states")
        logger.info(f"  - Keyword lookup: {len(self._keyword_lookup)} entries")
        logger.info(f"  - Phrase index: {len(self._phrase_index)} keys")
        logger.info(f"  - Pattern cache: {len(self._pattern_cache)} compiled patterns")
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment and emotional content"""
        if not self.sentiment_analyzer:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.warning(f"Sentiment analysis error: {str(e)}")
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
    
    def _analyze_context(self, text: str) -> Dict:
        """Enhanced context analysis with better detection logic"""
        context_score = 0.0
        indicators = []
        detailed_analysis = {}
        
        # 1. Caps lock analysis (fixed calculation)
        caps_analysis = self._analyze_caps_patterns(text)
        context_score += caps_analysis['score']
        indicators.extend(caps_analysis['indicators'])
        detailed_analysis['caps_analysis'] = caps_analysis
        
        # 2. Punctuation analysis (distinguish types)
        punct_analysis = self._analyze_punctuation_patterns(text)
        context_score += punct_analysis['score']
        indicators.extend(punct_analysis['indicators'])
        detailed_analysis['punctuation_analysis'] = punct_analysis
        
        # 3. Personal targeting analysis (improved)
        targeting_analysis = self._analyze_personal_targeting(text)
        context_score += targeting_analysis['score']
        indicators.extend(targeting_analysis['indicators'])
        detailed_analysis['targeting_analysis'] = targeting_analysis
        
        # 4. Command/imperative analysis
        command_analysis = self._analyze_command_patterns(text)
        context_score += command_analysis['score']
        indicators.extend(command_analysis['indicators'])
        detailed_analysis['command_analysis'] = command_analysis
        
        # 5. Repetition and emphasis patterns
        emphasis_analysis = self._analyze_emphasis_patterns(text)
        context_score += emphasis_analysis['score']
        indicators.extend(emphasis_analysis['indicators'])
        detailed_analysis['emphasis_analysis'] = emphasis_analysis
        
        # 6. Text structure analysis
        structure_analysis = self._analyze_text_structure(text)
        context_score += structure_analysis['score']
        indicators.extend(structure_analysis['indicators'])
        detailed_analysis['structure_analysis'] = structure_analysis
        
        return {
            'score': min(context_score, 1.0),
            'indicators': indicators,
            'detailed_analysis': detailed_analysis
        }
    
    def _analyze_caps_patterns(self, text: str) -> Dict:
        """Analyze capitalization patterns correctly"""
        if not text.strip():
            return {'score': 0.0, 'indicators': [], 'caps_ratio': 0.0, 'caps_sequences': 0}
        
        # Count alphabetic characters only for caps ratio
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return {'score': 0.0, 'indicators': [], 'caps_ratio': 0.0, 'caps_sequences': 0}
        
        caps_chars = [c for c in alpha_chars if c.isupper()]
        caps_ratio = len(caps_chars) / len(alpha_chars)
        
        # Find sequences of caps (4+ consecutive caps letters)
        caps_sequences = len(re.findall(r'[A-Z]{4,}', text))
        
        score = 0.0
        indicators = []
        
        # Score based on caps ratio
        if caps_ratio > 0.7:  # More than 70% caps
            score += 0.25
            indicators.append('excessive_caps')
        elif caps_ratio > 0.5:  # More than 50% caps
            score += 0.15
            indicators.append('high_caps')
        elif caps_ratio > 0.3:  # More than 30% caps
            score += 0.1
            indicators.append('moderate_caps')
        
        # Additional score for caps sequences
        if caps_sequences > 2:
            score += 0.15
            indicators.append('multiple_caps_sequences')
        elif caps_sequences > 0:
            score += 0.1
            indicators.append('caps_sequence')
        
        return {
            'score': score,
            'indicators': indicators,
            'caps_ratio': caps_ratio,
            'caps_sequences': caps_sequences
        }
    
    def _analyze_punctuation_patterns(self, text: str) -> Dict:
        """Analyze punctuation for aggression vs normal emphasis"""
        score = 0.0
        indicators = []
        
        # Count different types of punctuation
        excessive_exclamation = len(re.findall(r'[!]{2,}', text))
        excessive_question = len(re.findall(r'[?]{2,}', text))
        mixed_punct = len(re.findall(r'[?!]{3,}', text))
        ellipsis_abuse = len(re.findall(r'[.]{4,}', text))  # More than normal ellipsis
        
        # Distinguish between normal emphasis and aggressive punctuation
        normal_exclamation = len(re.findall(r'(?<![!])[!](?![!])', text))  # Single !
        normal_question = len(re.findall(r'(?<![?])[?](?![?])', text))    # Single ?
        
        text_length = len(text.split())
        
        # Aggressive punctuation patterns
        if excessive_exclamation > 0:
            # More weight for very excessive punctuation
            if excessive_exclamation > 2:
                score += 0.2
                indicators.append('very_excessive_exclamation')
            else:
                score += 0.1
                indicators.append('excessive_exclamation')
        
        if mixed_punct > 0:
            score += 0.15
            indicators.append('mixed_aggressive_punctuation')
        
        if ellipsis_abuse > 0:
            score += 0.05
            indicators.append('ellipsis_abuse')
        
        # Context-aware scoring - high punctuation density in short messages
        if text_length < 10:  # Short message
            punct_density = (excessive_exclamation + excessive_question + mixed_punct) / max(text_length, 1)
            if punct_density > 0.5:  # More than 0.5 aggressive punct per word
                score += 0.1
                indicators.append('high_punctuation_density')
        
        return {
            'score': score,
            'indicators': indicators,
            'excessive_exclamation': excessive_exclamation,
            'excessive_question': excessive_question,
            'mixed_punct': mixed_punct
        }
    
    def _analyze_personal_targeting(self, text: str) -> Dict:
        """Improved personal targeting detection with context awareness"""
        score = 0.0
        indicators = []
        
        text_lower = text.lower()
        
        # Different types of personal targeting
        second_person = re.findall(r'\b(you|your|yourself|you\'re|you\'ll|you\'ve)\b', text_lower)
        direct_address = re.findall(r'\b(hey you|listen you|you there)\b', text_lower)
        possessive_attacks = re.findall(r'\byour\s+(face|mother|family|life|problem)\b', text_lower)
        
        # Context-aware analysis
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        word_count = len(text.split())
        
        # Calculate targeting density
        if word_count > 0:
            targeting_density = len(second_person) / word_count
            
            # Adjust thresholds based on message length
            if word_count < 10:  # Short message
                high_threshold = 0.4
                medium_threshold = 0.2
            else:  # Longer message
                high_threshold = 0.3
                medium_threshold = 0.15
            
            if targeting_density > high_threshold:
                score += 0.25
                indicators.append('high_personal_targeting')
            elif targeting_density > medium_threshold:
                score += 0.15
                indicators.append('moderate_personal_targeting')
        
        # Direct aggressive addressing
        if direct_address:
            score += 0.2
            indicators.append('direct_aggressive_address')
        
        # Possessive attacks ("your mother", "your face", etc.)
        if possessive_attacks:
            score += 0.3
            indicators.append('possessive_attack_pattern')
        
        # Multiple targeting in short bursts
        if len(second_person) > 3 and sentence_count <= 2:
            score += 0.1
            indicators.append('concentrated_targeting')
        
        return {
            'score': score,
            'indicators': indicators,
            'targeting_density': len(second_person) / max(word_count, 1),
            'second_person_count': len(second_person),
            'possessive_attacks': len(possessive_attacks)
        }
    
    def _analyze_command_patterns(self, text: str) -> Dict:
        """Analyze command/imperative patterns that suggest aggression"""
        score = 0.0
        indicators = []
        
        text_lower = text.lower()
        
        # Aggressive imperatives
        aggressive_commands = re.findall(
            r'\b(shut up|go away|get lost|leave me alone|stop it|quit it|go die|kill yourself)\b', 
            text_lower
        )
        
        # Demanding phrases
        demands = re.findall(
            r'\b(you should|you must|you need to|you have to|you better)\s+\w+', 
            text_lower
        )
        
        # Threatening imperatives
        threats = re.findall(
            r'\b(i will|i\'ll|gonna|going to)\s+(hurt|kill|destroy|get|beat)\b',
            text_lower
        )
        
        if aggressive_commands:
            score += 0.4  # High score for explicit aggressive commands
            indicators.append('aggressive_commands')
        
        if demands:
            score += 0.2
            indicators.append('demanding_language')
        
        if threats:
            score += 0.5  # Very high score for threats
            indicators.append('threatening_imperatives')
        
        return {
            'score': score,
            'indicators': indicators,
            'aggressive_commands': len(aggressive_commands),
            'demands': len(demands),
            'threats': len(threats)
        }
    
    def _analyze_emphasis_patterns(self, text: str) -> Dict:
        """Analyze repetition and emphasis that may indicate aggression"""
        score = 0.0
        indicators = []
        
        # Word repetition (same word repeated)
        words = text.lower().split()
        word_repetition = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:  # Ignore short words
                word_repetition += 1
        
        # Character repetition within words (handled by normalization, but count original)
        char_repetition = len(re.findall(r'(.)\1{3,}', text))  # 4+ repeated chars
        
        # Spacing abuse for emphasis
        spaced_words = len(re.findall(r'\b\w\s+\w\s+\w\b', text))  # s p a c e d
        
        if word_repetition > 1:
            score += 0.1
            indicators.append('word_repetition')
        
        if char_repetition > 2:
            score += 0.15
            indicators.append('excessive_character_repetition')
        elif char_repetition > 0:
            score += 0.05
            indicators.append('character_repetition')
        
        if spaced_words > 0:
            score += 0.1
            indicators.append('spaced_emphasis')
        
        return {
            'score': score,
            'indicators': indicators,
            'word_repetition': word_repetition,
            'char_repetition': char_repetition
        }
    
    def _analyze_text_structure(self, text: str) -> Dict:
        """Analyze overall text structure for aggressive patterns"""
        score = 0.0
        indicators = []
        
        # Very short aggressive messages
        words = text.split()
        word_count = len(words)
        
        if word_count <= 3 and any(word.upper() == word and len(word) > 2 for word in words):
            score += 0.1
            indicators.append('short_caps_message')
        
        # Lack of proper sentence structure in aggressive context
        sentences = re.split(r'[.!?]+', text.strip())
        if len(sentences) > 3 and all(len(s.split()) <= 2 for s in sentences if s.strip()):
            score += 0.05
            indicators.append('fragmented_aggressive_structure')
        
        # All caps words mixed with normal text (selective emphasis)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if len(caps_words) > 0 and len(caps_words) < word_count:  # Some but not all caps
            caps_ratio = len(caps_words) / word_count
            if 0.2 <= caps_ratio <= 0.6:  # Selective caps usage
                score += 0.1
                indicators.append('selective_caps_emphasis')
        
        return {
            'score': score,
            'indicators': indicators,
            'word_count': word_count,
            'caps_words': len(caps_words)
        }
    
    def _classify_intent(self, text: str) -> Dict:
        """Classify the intent behind the message"""
        intents = {
            'threat': 0.0,
            'insult': 0.0,
            'exclusion': 0.0,
            'harassment': 0.0,
            'encouragement_of_harm': 0.0
        }
        
        text_lower = text.lower()
        
        # Threat detection
        threat_patterns = [
            r'\b(kill|hurt|harm|destroy|end)\s+(you|yourself)\b',
            r'\bi\s+will\s+(hurt|kill|destroy)\b',
            r'\bwatch\s+out\b'
        ]
        
        for pattern in threat_patterns:
            if re.search(pattern, text_lower):
                intents['threat'] += 0.3
        
        # Intent detection
        insult_patterns = [
            r'\byou\s+are\s+(so\s+)?(stupid|ugly|worthless|pathetic)\b',
            r'\bsuch\s+a\s+(loser|idiot|freak)\b',
            r'\bi\s+hate\s+you\b',
            r'\bhate\s+you\b'
        ]
        
        for pattern in insult_patterns:
            if re.search(pattern, text_lower):
                intents['insult'] += 0.4
        
        # Social exclusion
        exclusion_patterns = [
            r'\bnobody\s+(likes|wants|cares)\s+(about\s+)?you\b',
            r'\byou\s+don\'t\s+belong\b'
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, text_lower):
                intents['exclusion'] += 0.4
        
        return intents
    
    async def _enhanced_openai_check(self, text: str, context_data: Dict) -> Dict:
        """Enhanced OpenAI analysis with context"""
        if not self.openai_client:
            return self._get_fallback_result()
        
        try:
            prompt = f"""
            Analyze this message for cyberbullying with enhanced context:
            
            Message: "{text}"
            
            Context Analysis:
            - Sentiment: {context_data.get('sentiment', {})}
            - Context Score: {context_data.get('context_score', 0)}
            - Intent Classification: {context_data.get('intent', {})}
            
            Provide a detailed JSON analysis with:
            {{
                "is_bullying": boolean,
                "confidence": float (0.0-1.0),
                "severity": "none|low|medium|high|critical",
                "primary_category": "threat|insult|exclusion|harassment|other",
                "detected_patterns": [list of detected harmful patterns],
                "risk_factors": [list of risk indicators],
                "reasoning": "detailed explanation",
                "recommended_action": "ignore|warn|block|escalate"
            }}
            
            Consider:
            1. Intent and context, not just keywords
            2. Severity of potential psychological harm
            3. Whether it's clearly directed at someone
            4. Cultural and linguistic nuances
            5. Potential for escalation
            """
            
            response = await self.openai_client.chat.completions.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert AI safety researcher specializing in cyberbullying detection. Provide accurate, nuanced analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(response_text)
                return self._validate_openai_result(result)
            except json.JSONDecodeError:
                return self._parse_fallback_response(response_text)
                
        except Exception as e:
            logger.error(f"Enhanced OpenAI check failed: {str(e)}")
            return self._get_fallback_result()
    
    def _validate_openai_result(self, result: Dict) -> Dict:
        """Validate and normalize OpenAI response"""
        validated = {
            'is_bullying': bool(result.get('is_bullying', False)),
            'confidence': max(0.0, min(1.0, float(result.get('confidence', 0.0)))),
            'severity': result.get('severity', 'none'),
            'primary_category': result.get('primary_category', 'other'),
            'detected_patterns': result.get('detected_patterns', []),
            'risk_factors': result.get('risk_factors', []),
            'reasoning': result.get('reasoning', ''),
            'recommended_action': result.get('recommended_action', 'ignore')
        }
        return validated
    
    def detect_bullying_enhanced(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """Enhanced detection with multiple AI techniques"""
        start_time = datetime.utcnow()
        self.stats['total_detections'] += 1
        
        if not text or not text.strip():
            return self._empty_text_result()
        
        # Step 1: Sentiment Analysis
        sentiment = self._analyze_sentiment(text)
        
        # Step 2: Context Analysis
        context = self._analyze_context(text)
        
        # Step 3: Intent Classification
        intent = self._classify_intent(text)
        
        # Step 4: Local Pattern Matching
        local_result = self._enhanced_local_check(text, sentiment, context, intent)
        
        # Step 5: OpenAI Analysis (if needed and available)
        # Only use OpenAI if: 1) It's configured, 2) Local confidence is low, 3) Local didn't detect bullying
        if (self.openai_client and 
            local_result['confidence'] < confidence_threshold and 
            not local_result['is_bullying']):
            import asyncio
            context_data = {
                'sentiment': sentiment,
                'context_score': context['score'],
                'intent': intent
            }
            openai_result = asyncio.run(self._enhanced_openai_check(text, context_data))
            # Only combine if OpenAI actually returned a meaningful result
            if openai_result.get('confidence', 0) > 0:
                final_result = self._combine_enhanced_results(local_result, openai_result)
            else:
                final_result = local_result
        else:
            final_result = local_result
        
        # Step 6: Post-processing and validation
        final_result = self._post_process_result(final_result, text)
        
        # Update statistics
        self._update_stats(final_result)
        
        # Add processing metadata
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        final_result['processing_time'] = processing_time
        final_result['timestamp'] = datetime.utcnow().isoformat()
        
        return final_result
    
    def _enhanced_local_check(self, text: str, sentiment: Dict, context: Dict, intent: Dict) -> Dict:
        """Enhanced local detection with context awareness"""
        text_lower = text.lower()
        words = self._extract_words(text_lower)
        
        detected_categories = []
        confidence_factors = []
        risk_indicators = []
        
        # Special handling for single words that need context
        context_required_words = ['die', 'kill', 'hurt', 'harm', 'end']
        is_single_word = len(words) == 1
        
        # Category-based detection
        for category, patterns in self.category_patterns.items():
            category_score = 0.0
            found_items = []
            
            # Check keywords
            for word in words:
                if word in patterns['keywords']:
                    # Skip context-required words if they appear alone
                    if is_single_word and word in context_required_words:
                        continue
                    
                    # Context-aware filtering for certain categories
                    if category == 'social_exclusion':
                        # For social exclusion, require additional context clues
                        # Don't trigger on polite phrases
                        if self._is_likely_polite_phrase(text_lower, words):
                            continue
                        # Require negative sentiment or other aggressive indicators
                        if sentiment['compound'] > -0.2 and context['score'] < 0.2:
                            continue
                    
                    if category == 'threats':
                        # For threats, be very careful about generic words like "you"
                        # Skip common polite words that might be in threats keyword list
                        if word in ['you', 'your', 'yourself'] and self._is_likely_polite_phrase(text_lower, words):
                            continue
                        # For single generic words, require strong negative context
                        if word in ['you', 'your', 'yourself'] and len(words) <= 3:
                            if sentiment['compound'] > -0.3 and context['score'] < 0.3:
                                continue
                    
                    found_items.append(word)
                    # Improved confidence boost - higher for direct matches
                    if category == 'direct_insults':
                        category_score += patterns['confidence_boost'] * 1.5  # Boost insults
                    else:
                        category_score += patterns['confidence_boost']
            
            # Use optimized phrase detection via Aho-Corasick automaton
            # This replaces the O(n²) phrase sequence checking
            phrase_matches = self.phrase_automaton.search(' '.join(words))
            for match in phrase_matches:
                if match['category'] == category:
                    found_items.append(match['pattern'])
                    # Use confidence boost with match strength weighting
                    category_score += match['confidence_boost']
            
            # Check regex patterns
            for pattern in patterns['patterns']:
                matches = pattern.findall(text_lower)
                if matches:
                    found_items.extend(matches)
                    category_score += patterns['confidence_boost'] * 2.0  # Higher boost for pattern matches
            
            if found_items:
                detected_categories.append({
                    'category': category,
                    'items': found_items,
                    'score': category_score,
                    'severity': patterns['severity']
                })
                confidence_factors.append(category_score)
        
        # Sentiment integration
        if sentiment['compound'] < -0.5:
            confidence_factors.append(0.2)
            risk_indicators.append('negative_sentiment')
        
        # Context integration
        if context['score'] > 0.3:
            confidence_factors.append(context['score'])
            risk_indicators.extend(context['indicators'])
        
        # Intent integration
        max_intent = max(intent.values()) if intent.values() else 0
        if max_intent > 0.3:
            confidence_factors.append(max_intent)
            dominant_intent = max(intent.items(), key=lambda x: x[1])
            risk_indicators.append(f'intent_{dominant_intent[0]}')
        
        # Calculate final confidence with proper normalization
        adjusted_confidence = self._calculate_normalized_confidence(
            confidence_factors, detected_categories, context['score'], 
            sentiment['compound'], max_intent, is_single_word
        )
        
        # Determine severity
        severity = self._determine_severity(detected_categories, intent)
        
        # Determine if bullying - smarter thresholds
        # For single words, require actual detected categories, not just sentiment
        if is_single_word:
            # Single words need to be in our database to be flagged
            is_bullying = len(detected_categories) > 0
        else:
            # For phrases, use normal detection logic
            is_bullying = (
                len(detected_categories) > 0 or 
                adjusted_confidence >= 0.3 or
                max_intent > 0.3
            )
        
        return {
            'is_bullying': is_bullying,
            'confidence': adjusted_confidence,
            'severity': severity if is_bullying else 'none',
            'detected_categories': detected_categories,
            'risk_indicators': risk_indicators,
            'sentiment_analysis': sentiment,
            'context_analysis': context,
            'intent_classification': intent,
            'detection_method': 'enhanced_local'
        }
    
    def _determine_severity(self, categories: List[Dict], intent: Dict) -> str:
        """Determine overall severity based on detected categories and intent"""
        if not categories:
            return 'none'
        
        severity_scores = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        max_severity = 0
        
        for category in categories:
            severity = category.get('severity', 'medium')
            score = severity_scores.get(severity, 1)
            max_severity = max(max_severity, score)
        
        # Check for critical intent patterns
        if intent.get('threat', 0) > 0.5 or intent.get('encouragement_of_harm', 0) > 0.5:
            max_severity = max(max_severity, 4)
        
        # Convert back to string
        for severity, score in severity_scores.items():
            if score == max_severity:
                return severity
        
        return 'medium'
    
    def _combine_enhanced_results(self, local: Dict, openai: Dict) -> Dict:
        """Combine local and OpenAI results with intelligent weighting"""
        # Dynamic weighting based on confidence
        local_conf = local.get('confidence', 0)
        openai_conf = openai.get('confidence', 0)
        
        # If OpenAI returned 0 confidence (likely a fallback), don't let it dilute local results
        if openai_conf == 0 and local_conf > 0:
            return local
        
        if local_conf > 0.8:
            local_weight = 0.7
            openai_weight = 0.3
        elif openai_conf > 0.8:
            local_weight = 0.3
            openai_weight = 0.7
        else:
            local_weight = 0.5
            openai_weight = 0.5
        
        combined_confidence = (local_conf * local_weight) + (openai_conf * openai_weight)
        
        # Combine detected categories
        all_categories = local.get('detected_categories', [])
        if 'primary_category' in openai and openai['primary_category'] != 'other':
            all_categories.append({
                'category': openai['primary_category'],
                'source': 'openai',
                'confidence': openai_conf
            })
        
        # Determine is_bullying more intelligently
        # If local detection found bullying with decent confidence, trust it
        is_bullying = local['is_bullying'] if local_conf >= 0.5 else (
            combined_confidence >= 0.5 or local['is_bullying'] or openai.get('is_bullying', False)
        )
        
        return {
            'is_bullying': is_bullying,
            'confidence': combined_confidence,
            'severity': self._combine_severity(local.get('severity'), openai.get('severity')),
            'detected_categories': all_categories,
            'risk_indicators': local.get('risk_indicators', []) + openai.get('risk_factors', []),
            'sentiment_analysis': local.get('sentiment_analysis', {}),
            'openai_reasoning': openai.get('reasoning', ''),
            'recommended_action': openai.get('recommended_action', 'warn'),
            'detection_method': 'enhanced_combined'
        }
    
    def _combine_severity(self, local_severity: str, openai_severity: str) -> str:
        """Combine severity levels intelligently"""
        severity_order = ['none', 'low', 'medium', 'high', 'critical']
        
        local_idx = severity_order.index(local_severity) if local_severity in severity_order else 0
        openai_idx = severity_order.index(openai_severity) if openai_severity in severity_order else 0
        
        return severity_order[max(local_idx, openai_idx)]
    
    def _update_stats(self, result: Dict):
        """Update enhanced statistics"""
        if result.get('is_bullying'):
            self.stats['bullying_detected'] += 1
        
        # Track confidence distribution
        conf_bucket = int(result.get('confidence', 0) * 10) * 10
        self.stats['confidence_distribution'][conf_bucket] += 1
        
        # Track categories
        for category in result.get('detected_categories', []):
            cat_name = category.get('category', 'unknown')
            self.stats['category_counts'][cat_name] += 1
        
        # Track severity
        severity = result.get('severity', 'none')
        self.stats['severity_distribution'][severity] += 1
    
    def get_enhanced_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        # Add compatibility fields for the old statistics format
        stats_result = {
            **self.stats,
            'detection_accuracy': self._calculate_accuracy(),
            'category_breakdown': dict(self.stats['category_counts']),
            'severity_breakdown': dict(self.stats['severity_distribution']),
            'confidence_distribution': dict(self.stats['confidence_distribution']),
            # Compatibility fields
            'local_detections': self.stats['total_detections'],
            'openai_detections': 0,  # Not tracked in enhanced detector
            'bullying_found': self.stats['bullying_detected']
        }
        return stats_result
    
    def _calculate_accuracy(self) -> float:
        """Calculate detection accuracy based on feedback"""
        total = self.stats['total_detections']
        if total == 0:
            return 0.0
        
        false_positives = self.stats.get('false_positives', 0)
        return max(0.0, (total - false_positives) / total)
    
    def _empty_text_result(self) -> Dict:
        """Return result for empty text"""
        return {
            'is_bullying': False,
            'confidence': 0.0,
            'severity': 'none',
            'detected_categories': [],
            'detection_method': 'empty_text'
        }
    
    def _get_fallback_result(self) -> Dict:
        """Fallback result when AI analysis fails"""
        return {
            'is_bullying': False,
            'confidence': 0.0,
            'severity': 'none',
            'detection_method': 'fallback',
            'error': 'AI analysis unavailable'
        }
    
    def _post_process_result(self, result: Dict, original_text: str) -> Dict:
        """Post-process results for final validation"""
        # Adjust confidence based on text length
        text_length = len(original_text.split())
        if text_length < 3 and result['confidence'] > 0.5:
            result['confidence'] *= 0.8  # Reduce confidence for very short texts
        
        # Add explanation
        result['explanation'] = self._generate_explanation(result)
        
        return result
    
    def _generate_explanation(self, result: Dict) -> str:
        """Generate human-readable explanation"""
        if not result.get('is_bullying'):
            return "No cyberbullying detected. The message appears to be safe."
        
        severity = result.get('severity', 'medium')
        confidence = result.get('confidence', 0)
        categories = result.get('detected_categories', [])
        
        explanation = f"Cyberbullying detected with {severity} severity ({confidence:.0%} confidence)."
        
        if categories:
            cat_names = [cat.get('category', 'unknown') for cat in categories]
            explanation += f" Detected categories: {', '.join(cat_names)}."
        
        action = result.get('recommended_action', 'warn')
        if action == 'block':
            explanation += " Immediate action recommended."
        elif action == 'escalate':
            explanation += " Consider escalating to human moderator."
        
        return explanation
    
    def _check_phrase_sequence(self, text: str, phrase: str) -> bool:
        """Check if a multi-word phrase appears in sequence in the text"""
        # Split phrase into words
        phrase_words = phrase.split()
        text_words = text.split()
        
        # Check if phrase appears in sequence (allowing up to 2 words in between)
        for i in range(len(text_words)):
            matched_words = 0
            j = i
            
            for phrase_word in phrase_words:
                # Look for the next word in the phrase (within next 3 positions)
                found = False
                for k in range(j, min(j + 3, len(text_words))):
                    if text_words[k] == phrase_word:
                        matched_words += 1
                        j = k + 1
                        found = True
                        break
                
                if not found:
                    break
            
            # If all words matched in sequence, phrase is found
            if matched_words == len(phrase_words):
                return True
        
        return False
    
    def _is_likely_polite_phrase(self, text: str, words: List[str]) -> bool:
        """Check if the text is likely a polite phrase that shouldn't be flagged"""
        polite_indicators = [
            'thank you', 'thanks', 'please', 'sorry', 'excuse me',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'nice to meet you', 'how are you', 'pleased to meet',
            'have a good day', 'take care', 'welcome', 'congratulations',
            'happy birthday', 'best wishes'
        ]
        
        # Check for polite phrases
        for phrase in polite_indicators:
            if phrase in text:
                return True
        
        # Additional context clues for politeness
        polite_words = ['please', 'thank', 'thanks', 'sorry', 'welcome', 'congratulations']
        if any(word in words for word in polite_words):
            return True
            
        return False
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text for analysis"""
        # Normalize text first
        normalized_text = self._normalize_text(text.lower())
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', normalized_text)
        return words
    
    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization with comprehensive coverage"""
        # Convert to lowercase and handle mixed case issues
        original_text = text
        text = text.lower()
        
        # First pass: Handle repeated character normalization
        text = self._normalize_repeated_characters(text)
        
        # Second pass: Handle common misspellings and variants FIRST
        text = self._normalize_common_variants(text)
        
        # Third pass: Handle comprehensive leet speak substitutions
        text = self._normalize_leet_speak(text)
        
        # Fourth pass: Handle mixed case obfuscation (based on original text)
        text = self._normalize_mixed_case_obfuscation(original_text, text)
        
        # Fifth pass: Clean up punctuation and spacing
        text = self._normalize_punctuation_and_spacing(text)
        
        return text.strip()
    
    def _normalize_repeated_characters(self, text: str) -> str:
        """Normalize repeated characters (e.g., 'stuuuupid' -> 'stupid')"""
        # Replace 3+ consecutive repeated characters with just 2
        # This handles cases like 'nooooo', 'hiiiii', 'whaaaaaat'
        normalized = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # For vowels, often single is better for word matching
        # But keep consonants doubled to maintain word structure
        vowel_pattern = r'([aeiou])\1+'
        normalized = re.sub(vowel_pattern, r'\1', normalized)
        
        return normalized
    
    def _normalize_common_variants(self, text: str) -> str:
        """Handle common misspellings and variants comprehensively"""
        # Extended variant mapping with more comprehensive coverage
        common_variants = {
            # Hindi/Urdu profanity variants
            'maderchood': 'madarchod', 'maderchod': 'madarchod', 
            'madarchood': 'madarchod', 'motherchod': 'madarchod',
            'm@d@rch0d': 'madarchod', 'm4d4rch0d': 'madarchod',
            'benchood': 'bhenchod', 'benchod': 'bhenchod', 'banchod': 'bhenchod',
            'b3nch0d': 'bhenchod', 'b@nch0d': 'bhenchod',
            'madrchod': 'madarchod', 'mdrchod': 'madarchod',
            
            # English profanity variants
            'fukk': 'fuck', 'fuuk': 'fuck', 'fuk': 'fuck', 'phuck': 'fuck',
            'f*ck': 'fuck', 'f**k': 'fuck', 'fck': 'fuck', 'fvck': 'fuck',
            'f4ck': 'fuck', 'fu2k': 'fuck', 'f@ck': 'fuck',
            
            'shyt': 'shit', 'sht': 'shit', 'sh1t': 'shit', '$hit': 'shit',
            'sh*t': 'shit', 'shiit': 'shit', 'shitt': 'shit',
            
            'btch': 'bitch', 'b1tch': 'bitch', 'b*tch': 'bitch', 'bi7ch': 'bitch',
            'biatch': 'bitch', 'biotch': 'bitch', 'b!tch': 'bitch',
            
            'a55': 'ass', 'a$$': 'ass', '@ss': 'ass', 'azz': 'ass',
            'as$': 'ass', 'a$s': 'ass', '4ss': 'ass',
            
            'azzhole': 'asshole', 'a$$hole': 'asshole', '@sshole': 'asshole',
            'assh0le': 'asshole', 'a55hole': 'asshole',
            
            # Common insults variants
            'id10t': 'idiot', '1d10t': 'idiot', 'idi0t': 'idiot',
            'st*pid': 'stupid', 'stup1d': 'stupid', '$tupid': 'stupid',
            'dum@ss': 'dumbass', 'dumb@ss': 'dumbass', 'dumb4ss': 'dumbass',
            
            # Other common substitutions
            'l0ser': 'loser', 'l05er': 'loser', 'lo$er': 'loser',
            'h8': 'hate', 'h8r': 'hater', 'h@te': 'hate',
            'kil': 'kill', 'k1ll': 'kill', 'k!ll': 'kill',
        }
        
        # Apply variant substitutions with word boundaries
        for variant, standard in common_variants.items():
            pattern = r'\b' + re.escape(variant) + r'\b'
            text = re.sub(pattern, standard, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_leet_speak(self, text: str) -> str:
        """Comprehensive leet speak normalization"""
        # Comprehensive leet speak mapping
        leet_substitutions = {
            '@': 'a', '4': 'a', '/-\\': 'a', '/\\': 'a',
            '8': 'b', '|3': 'b', '13': 'b',
            '(': 'c', '<': 'c', '{': 'c',
            '|)': 'd', '[)': 'd',
            '3': 'e', '€': 'e',
            '|=': 'f', 'ph': 'f',
            '6': 'g', '9': 'g', '&': 'g',
            '#': 'h', '|-|': 'h', '[-]': 'h',
            '1': 'i', '!': 'i', '|': 'i',
            '_|': 'j',
            '|<': 'k', '|(': 'k',
            '|_': 'l', '1': 'l',
            '|\/|': 'm', '/\/\\': 'm',
            '|\\|': 'n', '/\/': 'n',
            '0': 'o', '()': 'o', '[]': 'o',
            '|*': 'p', '|o': 'p',
            '9': 'q', '0_': 'q',
            '|2': 'r', '12': 'r',
            '5': 's', '$': 's', 'z': 's',
            '7': 't', '+': 't', '|-': 't',
            '|_|': 'u', 'v': 'u',
            '\/': 'v', '\\|/': 'v',
            '\/\/': 'w', 'vv': 'w',
            '><': 'x', ')(': 'x',
            'j': 'y', '`/': 'y',
            '2': 'z', '7_': 'z',
        }
        
        # Apply simple single-character substitutions first
        simple_subs = {'@': 'a', '4': 'a', '3': 'e', '1': 'i', '!': 'i', 
                      '0': 'o', '5': 's', '7': 't', '$': 's', '+': 't'}
        
        for leet, normal in simple_subs.items():
            text = text.replace(leet, normal)
        
        # Handle more complex multi-character leet patterns
        complex_patterns = {
            r'\|\|': 'll', r'\|\\|': 'n', r'\|\/\|': 'm',
            r'\/\/': 'w', r'><': 'x', r'\|<': 'k'
        }
        
        for pattern, replacement in complex_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_mixed_case_obfuscation(self, original_text: str, normalized_text: str) -> str:
        """Handle mixed case obfuscation patterns"""
        # Check for alternating case patterns that might be obfuscation
        # Like: "StUpId", "bItCh", "DuMbAsS"
        words = original_text.split()
        normalized_words = normalized_text.split()
        
        result_words = []
        
        for i, (orig_word, norm_word) in enumerate(zip(words, normalized_words)):
            if len(orig_word) > 3:  # Only check longer words
                # Count case changes
                case_changes = 0
                for j in range(1, len(orig_word)):
                    if orig_word[j].isupper() != orig_word[j-1].isupper():
                        case_changes += 1
                
                # If there are many case changes, it might be obfuscation
                if case_changes > len(orig_word) // 3:  # More than 1/3 of chars change case
                    result_words.append(norm_word.lower())
                else:
                    result_words.append(norm_word)
            else:
                result_words.append(norm_word)
        
        return ' '.join(result_words)
    
    def _normalize_punctuation_and_spacing(self, text: str) -> str:
        """Clean up punctuation and spacing"""
        # Normalize excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'[.]{3,}', '...', text)  # Multiple periods
        text = re.sub(r'[-]{2,}', '-', text)  # Multiple dashes
        text = re.sub(r'[*]{2,}', '*', text)  # Multiple asterisks
        
        # Handle punctuation used as separators or obfuscation
        text = re.sub(r'([a-z])[-_*@#$%^&+=|~`]([a-z])', r'\1\2', text)  # Remove separators between letters
        
        # Clean up excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing punctuation from words while preserving
        # letters and digits from all languages (Unicode-aware).
        words = text.split()
        cleaned_words = []
        for word in words:
            # Strip non-word characters at the edges but keep Unicode letters/digits
            cleaned_word = re.sub(r'^\W+|\W+$', '', word)
            if cleaned_word:  # Only add non-empty words
                cleaned_words.append(cleaned_word)
        
        return ' '.join(cleaned_words)
    
    def _parse_fallback_response(self, response_text: str) -> Dict:
        """Parse fallback response when JSON parsing fails"""
        # Simple keyword-based parsing as fallback
        is_bullying = any(word in response_text.lower() for word in ['bullying', 'harassment', 'threat'])
        
        return {
            'is_bullying': is_bullying,
            'confidence': 0.5 if is_bullying else 0.0,
            'severity': 'medium' if is_bullying else 'none',
            'detection_method': 'fallback_parsed',
            'raw_response': response_text
        }
    
    def add_bullying_words(self, words: List[str]) -> int:
        """Add new bullying words to the database"""
        if not words:
            return 0
        
        added_count = 0
        
        # Add words to the direct_insults category by default
        current_keywords = self.bullying_data.get('bullying_categories', {}).get('direct_insults', {}).get('keywords', [])
        
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in current_keywords:
                current_keywords.append(word_clean)
                added_count += 1
        
        # Update the category patterns
        if added_count > 0:
            self.bullying_data.setdefault('bullying_categories', {}).setdefault('direct_insults', {})['keywords'] = current_keywords
            self._preprocess_enhanced_data()  # Recompile patterns
            
            # Save to file if possible
            try:
                with open(self.data_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.bullying_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {added_count} new words to database")
            except Exception as e:
                logger.warning(f"Could not save updated database: {str(e)}")
        
        return added_count
    
    def _convert_legacy_data(self, legacy_data: Dict) -> Dict:
        """Convert legacy bullying_words.json format to new enhanced format"""
        # Check if this is the new languages-based format
        if 'languages' in legacy_data:
            return self._convert_languages_format(legacy_data)
        
        # Old format with direct bullying_words list
        bullying_words = legacy_data.get('bullying_words', [])
        bullying_phrases = legacy_data.get('bullying_phrases', [])
        
        logger.info(f"Converting {len(bullying_words)} words from legacy format")
        logger.debug(f"Sample words: {bullying_words[:5]}")
        
        # Categorize words based on severity and type
        threat_words = []
        insult_words = []
        profanity_words = []
        
        # Simple categorization based on content
        for word in bullying_words:
            word_lower = word.lower().strip()
            if any(threat in word_lower for threat in ['kill', 'die', 'hurt', 'harm', 'destroy', 'end']):
                threat_words.append(word_lower)
            elif any(profanity in word_lower for profanity in ['chod', 'fuck', 'bitch', 'ass', 'shit']):
                profanity_words.append(word_lower)
            else:
                insult_words.append(word_lower)
        
        logger.info(f"Categorized: {len(insult_words)} insults, {len(profanity_words)} profanity, {len(threat_words)} threats")
        
        return {
            'bullying_categories': {
                'direct_insults': {
                    'keywords': insult_words + profanity_words,  # Combine insults and profanity
                    'patterns': [r'\byou\s+are\s+(so\s+)?(stupid|ugly|worthless)', r'\bgo\s+die\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.4
                },
                'threats': {
                    'keywords': threat_words,
                    'patterns': [r'\bkill\s+yourself\b', r'\bi\s+will\s+hurt\s+you\b'],
                    'severity': 'high',
                    'confidence_boost': 0.6
                },
                'social_exclusion': {
                    'keywords': ['nobody likes you', 'friendless', 'alone', 'outcast'],
                    'patterns': [r'\bnobody\s+(likes|wants)\s+you\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.4
                },
                'harassment': {
                    'keywords': ['freak', 'weirdo', 'creep', 'stalker'],
                    'patterns': [r'\byou\s+(creep|freak)\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.35
                }
            },
            'bullying_phrases': bullying_phrases,
            'severity_patterns': {
                'high': [r'\b(kill\s+yourself|end\s+your\s+life|go\s+die)\b'],
                'medium': [r'\b(stupid|ugly|worthless|pathetic)\b'],
                'low': [r'\b(weird|annoying|dumb)\b']
            },
            'context_indicators': {
                'aggressive_tone': ['!!!', 'CAPS LOCK', 'excessive punctuation'],
                'targeting': ['you', 'your', 'yourself'],
                'intent_markers': ['should', 'need to', 'must', 'have to']
            },
            'emotional_markers': {
                'anger': ['hate', 'angry', 'furious', 'rage'],
                'disgust': ['disgusting', 'gross', 'sick', 'revolting'],
                'contempt': ['pathetic', 'worthless', 'useless', 'waste']
            }
        }
    
    def _convert_languages_format(self, languages_data: Dict) -> Dict:
        """Convert languages-based format to enhanced categorized format"""
        all_words = []
        all_phrases = []
        
        # Extract all words and phrases from all languages
        languages = languages_data.get('languages', {})
        
        for lang_name, lang_data in languages.items():
            # Get single words
            single_words = lang_data.get('single_words', [])
            all_words.extend(single_words)
            
            # Get multi-word phrases
            multi_word_phrases = lang_data.get('multi_word_phrases', [])
            all_phrases.extend(multi_word_phrases)
            
            logger.info(f"Loading {len(single_words)} words and {len(multi_word_phrases)} phrases from {lang_name}")
        
        logger.info(f"Total loaded: {len(all_words)} words and {len(all_phrases)} phrases from all languages")
        
        # Categorize words based on content and severity
        threat_words = []
        insult_words = []
        profanity_words = []
        exclusion_phrases = []
        
        # Categorize single words
        for word in all_words:
            word_lower = word.lower().strip()
            if any(threat in word_lower for threat in ['kill', 'die', 'hurt', 'harm', 'destroy', 'end']):
                threat_words.append(word_lower)
            elif any(profanity in word_lower for profanity in ['chod', 'fuck', 'bitch', 'ass', 'shit', 'madarchod', 'bhenchod']):
                profanity_words.append(word_lower)
            else:
                insult_words.append(word_lower)
        
        # Categorize phrases
        threat_phrases = []
        for phrase in all_phrases:
            phrase_lower = phrase.lower().strip()
            if any(pattern in phrase_lower for pattern in ['kill yourself', 'go die', 'you should die', 'kys']):
                threat_phrases.append(phrase_lower)
            elif any(pattern in phrase_lower for pattern in ['nobody likes you', 'nobody cares', 'you have no']):
                exclusion_phrases.append(phrase_lower)
            else:
                # Add other phrases to insult category as multi-word phrases
                exclusion_phrases.append(phrase_lower)
        
        logger.info(f"Categorized: {len(insult_words)} insults, {len(profanity_words)} profanity, {len(threat_words)} threat words")
        logger.info(f"Phrase categories: {len(threat_phrases)} threats, {len(exclusion_phrases)} exclusion/insult phrases")
        
        return {
            'bullying_categories': {
                'direct_insults': {
                    'keywords': insult_words + profanity_words,  # Combine insults and profanity
                    'multi_word_phrases': [p for p in exclusion_phrases if not any(t in p for t in ['nobody', 'kill', 'die'])],  # Non-threat phrases
                    'patterns': [r'\byou\s+are\s+(so\s+)?(stupid|ugly|worthless)', r'\bgo\s+die\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.4
                },
                'threats': {
                    'keywords': threat_words,
                    'multi_word_phrases': threat_phrases,
                    'patterns': [r'\bkill\s+yourself\b', r'\bi\s+will\s+hurt\s+you\b', r'\bkys\b'],
                    'severity': 'high',
                    'confidence_boost': 0.6
                },
                'social_exclusion': {
                    'keywords': ['friendless', 'alone', 'outcast', 'nobody'],
                    'multi_word_phrases': [p for p in exclusion_phrases if any(t in p for t in ['nobody', 'alone', 'friendless'])],
                    'patterns': [r'\bnobody\s+(likes|wants|cares)\s+(about\s+)?you\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.4
                },
                'harassment': {
                    'keywords': ['freak', 'weirdo', 'creep', 'stalker'],
                    'patterns': [r'\byou\s+(creep|freak)\b'],
                    'severity': 'medium',
                    'confidence_boost': 0.35
                }
            },
            'severity_patterns': {
                'high': [r'\b(kill\s+yourself|end\s+your\s+life|go\s+die|kys)\b'],
                'medium': [r'\b(stupid|ugly|worthless|pathetic|madarchod|bhenchod|chutiya|gandu)\b'],
                'low': [r'\b(weird|annoying|dumb)\b']
            },
            'context_indicators': {
                'aggressive_tone': ['!!!', 'CAPS LOCK', 'excessive punctuation'],
                'targeting': ['you', 'your', 'yourself'],
                'intent_markers': ['should', 'need to', 'must', 'have to']
            },
            'emotional_markers': {
                'anger': ['hate', 'angry', 'furious', 'rage'],
                'disgust': ['disgusting', 'gross', 'sick', 'revolting'],
                'contempt': ['pathetic', 'worthless', 'useless', 'waste']
            }
        }
    
    def _detect_database_format(self, data: Dict) -> str:
        """Reliably detect database format type"""
        try:
            # Check for enhanced format - must have bullying_categories
            if 'bullying_categories' in data and isinstance(data['bullying_categories'], dict):
                # Further validate it has the expected structure
                categories = data['bullying_categories']
                if any('keywords' in cat_data or 'patterns' in cat_data for cat_data in categories.values() if isinstance(cat_data, dict)):
                    return 'enhanced'
            
            # Check for legacy languages format
            if 'languages' in data and isinstance(data['languages'], dict):
                return 'legacy_languages'
            
            # Check for legacy words format
            if 'bullying_words' in data and isinstance(data['bullying_words'], list):
                return 'legacy_words'
            
            # Unknown format
            return 'unknown'
            
        except Exception as e:
            logger.warning(f"Error detecting database format: {e}")
            return 'unknown'
    
    def _validate_enhanced_format(self, data: Dict) -> Dict:
        """Validate and fill missing fields in enhanced format"""
        try:
            # Required top-level fields
            required_fields = {
                'bullying_categories': {},
                'severity_patterns': {
                    'high': [r'\b(kill\s+yourself|end\s+your\s+life|go\s+die)\b'],
                    'medium': [r'\b(stupid|ugly|worthless|pathetic)\b'],
                    'low': [r'\b(weird|annoying|dumb)\b']
                },
                'context_indicators': {
                    'aggressive_tone': ['!!!', 'CAPS LOCK', 'excessive punctuation'],
                    'targeting': ['you', 'your', 'yourself'],
                    'intent_markers': ['should', 'need to', 'must', 'have to']
                },
                'emotional_markers': {
                    'anger': ['hate', 'angry', 'furious', 'rage'],
                    'disgust': ['disgusting', 'gross', 'sick', 'revolting'],
                    'contempt': ['pathetic', 'worthless', 'useless', 'waste']
                }
            }
            
            # Fill missing top-level fields
            for field, default_value in required_fields.items():
                if field not in data:
                    data[field] = default_value
                    logger.info(f"Added missing field: {field}")
            
            # Validate and fix category structure
            categories = data['bullying_categories']
            for category_name, category_data in categories.items():
                if not isinstance(category_data, dict):
                    logger.warning(f"Invalid category data for {category_name}, skipping")
                    continue
                
                # Ensure required category fields
                category_defaults = {
                    'keywords': [],
                    'patterns': [],
                    'severity': 'medium',
                    'confidence_boost': 0.3
                }
                
                for field, default_value in category_defaults.items():
                    if field not in category_data:
                        category_data[field] = default_value
            
            logger.info("Enhanced format validated and fixed")
            return data
            
        except Exception as e:
            logger.error(f"Error validating enhanced format: {e}")
            return data  # Return as-is if validation fails
    
    def _save_database_safely(self, data: Dict) -> bool:
        """Safely save database with backup and error handling"""
        try:
            # Create backup of existing file if it exists
            if os.path.exists(self.data_file_path):
                backup_path = f"{self.data_file_path}.backup"
                try:
                    import shutil
                    shutil.copy2(self.data_file_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Save the new data
            with open(self.data_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("Database saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
    
    def _calculate_normalized_confidence(
        self, 
        confidence_factors: List[float], 
        detected_categories: List[Dict],
        context_score: float,
        sentiment_compound: float,
        max_intent: float,
        is_single_word: bool
    ) -> float:
        """Calculate properly normalized confidence with bias correction"""
        if not confidence_factors:
            return 0.0
        
        # Base confidence from category matches
        category_confidence = 0.0
        if detected_categories:
            # Use weighted average instead of sum to prevent unbounded growth
            category_scores = [cat['score'] for cat in detected_categories]
            if category_scores:
                # Weight higher-scoring categories more
                weights = [score / sum(category_scores) for score in category_scores]
                category_confidence = sum(score * weight for score, weight in zip(category_scores, weights))
                
                # Apply diminishing returns for multiple categories
                num_categories = len(detected_categories)
                if num_categories > 1:
                    # Boost confidence for multiple categories, but with diminishing returns
                    multi_category_boost = 1.0 + (0.1 * num_categories) / (1.0 + 0.05 * num_categories)
                    category_confidence *= multi_category_boost
        
        # Contextual confidence adjustments
        contextual_confidence = 0.0
        
        # Sentiment contribution (normalized)
        if sentiment_compound < -0.3:
            sentiment_weight = min(abs(sentiment_compound) * 0.3, 0.2)  # Max 0.2 boost
            contextual_confidence += sentiment_weight
        
        # Context score contribution (normalized)
        if context_score > 0.2:
            context_weight = min(context_score * 0.4, 0.25)  # Max 0.25 boost  
            contextual_confidence += context_weight
        
        # Intent contribution (normalized)
        if max_intent > 0.2:
            intent_weight = min(max_intent * 0.5, 0.3)  # Max 0.3 boost
            contextual_confidence += intent_weight
        
        # Combine category and contextual confidence
        base_confidence = category_confidence + contextual_confidence
        
        # Single word penalty (reduce overconfidence in single words)
        if is_single_word:
            if not detected_categories:  # No category matches for single word
                base_confidence *= 0.3  # Heavy penalty
            else:
                base_confidence *= 0.7  # Moderate penalty even with matches
        
        # Apply ceiling and normalization
        normalized_confidence = min(base_confidence, 1.0)
        
        # Apply confidence calibration based on detection strength
        if normalized_confidence < 0.1:
            # Very low confidence - keep as is
            final_confidence = normalized_confidence
        elif normalized_confidence < 0.3:
            # Low confidence - slight boost if we have good evidence
            evidence_boost = 1.0
            if detected_categories and context_score > 0.3:
                evidence_boost = 1.1
            final_confidence = min(normalized_confidence * evidence_boost, 0.35)
        elif normalized_confidence < 0.7:
            # Medium confidence - apply slight smoothing
            final_confidence = normalized_confidence * 0.95
        else:
            # High confidence - ensure it's well justified
            if len(detected_categories) > 1 or (detected_categories and context_score > 0.4):
                final_confidence = normalized_confidence
            else:
                # Reduce overconfidence if evidence is weak
                final_confidence = min(normalized_confidence * 0.85, 0.8)
        
        return max(0.0, min(final_confidence, 1.0))
    
    def _generate_phrase_key(self, words: List[str]) -> str:
        """Generate a unique key for phrase indexing.
        
        Creates a normalized key that allows for efficient phrase lookup
        while being somewhat flexible to word order variations.
        """
        # Sort words for consistent key generation (allows some flexibility)
        sorted_words = sorted(words)
        # Join with a delimiter that won't appear in normalized words
        return '|'.join(sorted_words)
    
    def _precompile_normalization_patterns(self):
        """Pre-compile common normalization regex patterns for better performance."""
        # Pre-compile frequently used normalization patterns
        normalization_patterns = {
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'vowel_repetition': re.compile(r'([aeiou])\1+'),
            'excessive_exclamation': re.compile(r'[!]{2,}'),
            'excessive_question': re.compile(r'[?]{2,}'),
            'excessive_periods': re.compile(r'[.]{3,}'),
            'multiple_dashes': re.compile(r'[-]{2,}'),
            'multiple_asterisks': re.compile(r'[*]{2,}'),
            'caps_sequences': re.compile(r'[A-Z]{4,}'),
            'letter_separators': re.compile(r'([a-z])[-_*@#$%^&+=|~`]([a-z])'),
            'excessive_spaces': re.compile(r'\s+'),
            'word_punctuation': re.compile(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$'),
            'mixed_punct': re.compile(r'[?!]{3,}'),
            'ellipsis_abuse': re.compile(r'[.]{4,}'),
            'char_repetition': re.compile(r'(.)\1{3,}'),
            'spaced_words': re.compile(r'\b\w\s+\w\s+\w\b')
        }
        
        # Cache these patterns for reuse
        for pattern_name, pattern in normalization_patterns.items():
            cache_key = f'norm_{pattern_name}'
            self._pattern_cache[cache_key] = pattern
        
        logger.info(f"Pre-compiled {len(normalization_patterns)} normalization patterns")
    
    def _get_cached_pattern(self, pattern_name: str) -> Optional[re.Pattern]:
        """Get a cached regex pattern by name."""
        cache_key = f'norm_{pattern_name}'
        return self._pattern_cache.get(cache_key)
    
    def _cached_text_normalize(self, text: str) -> str:
        """Optimized text normalization using LRU cache."""
        # Check cache first using LRU cache
        cached_result = self._normalized_text_cache.get(text)
        if cached_result is not None:
            return cached_result
        
        # Normalize using cached patterns
        normalized = self._normalize_text(text)
        
        # Cache result using LRU cache (automatically handles size limits)
        self._normalized_text_cache.put(text, normalized)
        
        return normalized
    
    def _quick_keyword_lookup(self, word: str) -> Optional[Dict]:
        """O(1) keyword lookup using the cached keyword dictionary."""
        return self._keyword_lookup.get(word.lower())
    
    def _indexed_phrase_lookup(self, words: List[str]) -> List[Dict]:
        """Quick phrase lookup using the phrase index."""
        phrase_key = self._generate_phrase_key(words)
        return self._phrase_index.get(phrase_key, [])
    
    def get_cache_statistics(self) -> Dict:
        """Get statistics about the caching system performance."""
        cache_stats = self._normalized_text_cache.stats()
        return {
            'pattern_cache_size': len(self._pattern_cache),
            'normalized_text_cache_size': cache_stats['size'],
            'normalized_text_cache_max_size': cache_stats['max_size'],
            'normalized_text_cache_utilization': cache_stats['utilization'],
            'keyword_lookup_size': len(self._keyword_lookup),
            'phrase_index_size': len(self._phrase_index),
            'total_cache_entries': (len(self._pattern_cache) + 
                                  cache_stats['size'] + 
                                  len(self._keyword_lookup) + 
                                  len(self._phrase_index))
        }
    
    def clear_caches(self):
        """Clear all caches to free memory."""
        self._normalized_text_cache.clear()
        # Don't clear pattern cache or keyword/phrase indices as they're structural
        logger.info("Cleared text normalization cache")
