"""
Advanced Multilingual Cyberbullying Detection System
Features: Async support, dependency injection, caching, ML integration
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import unicodedata
from collections import defaultdict, Counter
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for bullying detection"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        weights = {
            self.NONE: 0.0,
            self.LOW: 0.25,
            self.MEDIUM: 0.5,
            self.HIGH: 0.75,
            self.CRITICAL: 1.0
        }
        return weights[self]


class DetectionType(Enum):
    """Types of detection methods"""
    SINGLE_WORD = "single_word"
    PHRASE = "phrase"
    PATTERN = "pattern"
    CONTEXT = "context"
    ML_PREDICTION = "ml_prediction"


@dataclass
class Detection:
    """Represents a single detection result"""
    type: DetectionType
    value: str
    language: str
    severity: Severity
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    is_bullying: bool
    confidence: float
    severity: Severity
    detections: List[Detection]
    languages: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigurationManager:
    """Manages all configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files"""
        configs = [
            "language_detection.json",
            "text_normalization.json", 
            "detection_features.json"
        ]
        
        for config_file in configs:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_name = config_file.replace('.json', '')
                    self._cache[config_name] = json.load(f)
            else:
                logger.warning(f"Configuration file not found: {config_path}")
    
    def get(self, config_name: str, key: str = None) -> Any:
        """Get configuration value"""
        config = self._cache.get(config_name, {})
        if key:
            return config.get(key)
        return config
    
    def reload(self):
        """Reload all configurations"""
        self._cache.clear()
        self._load_all_configs()


class TextNormalizer:
    """Advanced text normalization with configuration support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.leet_mappings = config.get("leet_speak_mappings", {})
        self.misspellings = config.get("word_variations", {}).get("common_misspellings", {})
        self.punctuation_rules = config.get("punctuation_rules", {})
    
    @lru_cache(maxsize=1000)
    def normalize(self, text: str) -> str:
        """Normalize text with caching"""
        # Convert to lowercase
        text = text.lower()
        
        # Apply misspelling corrections
        text = self._correct_misspellings(text)
        
        # Apply leet speak conversions
        text = self._convert_leet_speak(text)
        
        # Handle elongated characters
        text = self._handle_elongation(text)
        
        # Clean punctuation
        text = self._clean_punctuation(text)
        
        return text.strip()
    
    def _correct_misspellings(self, text: str) -> str:
        """Correct common misspellings"""
        for misspelling, correction in self.misspellings.items():
            text = re.sub(r'\b' + re.escape(misspelling) + r'\b', correction, text, flags=re.IGNORECASE)
        return text
    
    def _convert_leet_speak(self, text: str) -> str:
        """Convert leet speak to normal text"""
        # Apply common mappings
        for leet, normal in self.leet_mappings.get("common", {}).items():
            text = text.replace(leet, normal)
        
        # Apply advanced mappings
        for leet, normal in self.leet_mappings.get("advanced", {}).items():
            text = text.replace(leet, normal)
        
        return text
    
    def _handle_elongation(self, text: str) -> str:
        """Handle elongated words (e.g., 'stuuuupid' -> 'stupid')"""
        threshold = self.config.get("word_variations", {}).get("elongation_threshold", 3)
        pattern = r'(.)\1{' + str(threshold - 1) + ',}'
        return re.sub(pattern, r'\1', text)
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean excessive punctuation"""
        rules = self.punctuation_rules.get("remove_excessive", {})
        
        for punct, rule in rules.items():
            max_count = rule.get("max", 1)
            replacement = rule.get("replacement", punct)
            pattern = re.escape(punct) + '{' + str(max_count + 1) + ',}'
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Advanced tokenization"""
        # Basic word tokenization
        words = re.findall(r'\b[\w]+\b', text)
        
        # Filter by length
        min_len = self.config.get("tokenization", {}).get("min_token_length", 2)
        max_len = self.config.get("tokenization", {}).get("max_token_length", 50)
        
        return [w for w in words if min_len <= len(w) <= max_len]


class LanguageDetector:
    """Language detection with configuration support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("language_indicators", {})
        self.rules = config.get("detection_rules", {})
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for script detection"""
        self.script_patterns = {}
        for lang, data in self.config.items():
            if "script_ranges" in data:
                patterns = []
                for range_str in data["script_ranges"]:
                    # Unescape unicode ranges
                    range_str = range_str.encode().decode('unicode_escape')
                    patterns.append(f"[{range_str}]")
                
                if patterns:
                    self.script_patterns[lang] = re.compile('|'.join(patterns))
    
    def detect_languages(self, text: str) -> List[str]:
        """Detect languages present in text"""
        detected = set()
        words = text.lower().split()
        
        # Script-based detection
        if self.rules.get("script_detection_enabled", True):
            for lang, pattern in self.script_patterns.items():
                if pattern.search(text):
                    detected.add(lang)
        
        # Word-based detection
        for lang, data in self.config.items():
            word_indicators = set(data.get("word_indicators", []))
            phrase_indicators = data.get("phrase_indicators", [])
            
            # Check word indicators
            matching_words = sum(1 for word in words if word in word_indicators)
            if matching_words >= self.rules.get("min_words_for_detection", 2):
                detected.add(lang)
                continue
            
            # Check phrase indicators
            text_lower = text.lower()
            for phrase in phrase_indicators:
                if phrase in text_lower:
                    detected.add(lang)
                    break
        
        # Add default language if needed
        if not detected and self.rules.get("default_language"):
            detected.add(self.rules["default_language"])
        
        return list(detected)


class PhraseDetector:
    """Advanced phrase detection with flexibility"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flexibility = config.get("flexibility", {})
        self.scoring = config.get("scoring", {})
    
    def find_phrase(self, words: List[str], phrase_words: List[str]) -> Optional[Tuple[bool, float]]:
        """Find phrase in word list with flexible matching"""
        max_between = self.flexibility.get("max_words_between", 2)
        phrase_len = len(phrase_words)
        
        if len(words) < phrase_len:
            return None
        
        # Try to find the phrase
        for i in range(len(words) - phrase_len + 1):
            matched_positions = []
            current_pos = i
            
            for j, phrase_word in enumerate(phrase_words):
                found = False
                search_limit = min(current_pos + max_between + 1, len(words))
                
                for k in range(current_pos, search_limit):
                    if words[k] == phrase_word:
                        matched_positions.append(k)
                        current_pos = k + 1
                        found = True
                        break
                
                if not found:
                    break
            
            if len(matched_positions) == phrase_len:
                # Calculate score based on compactness
                span = matched_positions[-1] - matched_positions[0] + 1
                gaps = span - phrase_len
                
                if gaps == 0:
                    score = self.scoring.get("exact_match", 1.0)
                elif gaps == 1:
                    score = self.scoring.get("one_word_between", 0.8)
                else:
                    score = self.scoring.get("two_words_between", 0.6)
                
                return True, score
        
        return None


class ContextAnalyzer:
    """Analyze context and aggression indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggressive_indicators = config.get("aggressive_indicators", {})
        self.targeting = config.get("targeting_indicators", {})
        self.commands = config.get("command_patterns", [])
    
    def analyze(self, text: str, words: List[str]) -> Dict[str, Any]:
        """Analyze text context"""
        results = {
            "aggression_score": 0.0,
            "targeting_score": 0.0,
            "command_score": 0.0,
            "indicators": []
        }
        
        # Check punctuation patterns
        for pattern_info in self.aggressive_indicators.get("punctuation_patterns", []):
            pattern = re.compile(pattern_info["pattern"])
            if pattern.search(text):
                results["aggression_score"] += pattern_info["weight"]
                results["indicators"].append(pattern_info["name"])
        
        # Check caps patterns
        caps_count = sum(1 for c in text if c.isupper())
        if len(text) > 0:
            caps_ratio = caps_count / len(text)
            for pattern_info in self.aggressive_indicators.get("caps_patterns", []):
                if caps_ratio >= pattern_info.get("min_percentage", 0.5):
                    results["aggression_score"] += pattern_info["weight"]
                    results["indicators"].append(pattern_info["name"])
        
        # Check targeting
        pronouns = set(self.targeting.get("personal_pronouns", []))
        pronoun_count = sum(1 for word in words if word in pronouns)
        if pronoun_count >= self.targeting.get("targeting_threshold", 2):
            results["targeting_score"] = self.targeting.get("weight", 0.2)
            results["indicators"].append("personal_targeting")
        
        # Check commands
        for command_group in self.commands:
            command_words = set(command_group["words"])
            if any(word in command_words for word in words):
                results["command_score"] += command_group["weight"]
                results["indicators"].append("command_structure")
        
        return results


class BullyingDatabaseManager:
    """Manages the bullying words database"""
    
    def __init__(self, database_path: str = "data/bullying_words.json"):
        self.database_path = Path(database_path)
        self.data = self._load_database()
        self._preprocess_data()
    
    def _load_database(self) -> Dict[str, Any]:
        """Load the database file"""
        if self.database_path.exists():
            with open(self.database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Database not found: {self.database_path}")
            return {"languages": {}, "metadata": {}}
    
    def _preprocess_data(self):
        """Preprocess data for efficient lookup"""
        self.processed_data = {}
        
        for language, lang_data in self.data.get("languages", {}).items():
            # Create sets for fast lookup
            single_words = set(lang_data.get("single_words", []))
            
            # Process phrases
            phrases = []
            for phrase in lang_data.get("multi_word_phrases", []):
                phrases.append({
                    "text": phrase,
                    "words": phrase.split(),
                    "severity": Severity(lang_data.get("severity_map", {}).get(phrase, "medium"))
                })
            
            # Compile patterns
            patterns = []
            for pattern_str in lang_data.get("patterns", []):
                try:
                    patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid pattern in {language}: {pattern_str} - {e}")
            
            self.processed_data[language] = {
                "single_words": single_words,
                "phrases": phrases,
                "patterns": patterns,
                "severity_map": lang_data.get("severity_map", {})
            }
    
    def get_language_data(self, language: str) -> Dict[str, Any]:
        """Get processed data for a language"""
        return self.processed_data.get(language, {})
    
    def add_words(self, language: str, words: List[str], word_type: str = "single") -> bool:
        """Add new words to the database"""
        if language not in self.data["languages"]:
            self.data["languages"][language] = {
                "single_words": [],
                "multi_word_phrases": [],
                "patterns": [],
                "severity_map": {}
            }
        
        target_list = "single_words" if word_type == "single" else "multi_word_phrases"
        current_items = set(self.data["languages"][language].get(target_list, []))
        
        added = False
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in current_items:
                self.data["languages"][language][target_list].append(word_clean)
                added = True
        
        if added:
            self._save_database()
            self._preprocess_data()
        
        return added
    
    def _save_database(self):
        """Save the database"""
        self.data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


class CacheManager:
    """Manages caching for performance optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.ttl = timedelta(seconds=config.get("ttl_seconds", 3600))
        self.max_size = config.get("max_size", 10000)
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.enabled:
            return None
        
        if key in self._cache:
            timestamp = self._timestamps.get(key)
            if timestamp and datetime.utcnow() - timestamp < self.ttl:
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        if not self.enabled:
            return
        
        # Check size limit
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = datetime.utcnow()
    
    def create_key(self, text: str) -> str:
        """Create cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()


class AdvancedBullyingDetector:
    """Main detector class with all advanced features"""
    
    def __init__(self, config_dir: str = "config", database_path: str = "data/bullying_words.json"):
        # Initialize configuration
        self.config_manager = ConfigurationManager(config_dir)
        
        # Initialize components
        self.text_normalizer = TextNormalizer(
            self.config_manager.get("text_normalization")
        )
        self.language_detector = LanguageDetector(
            self.config_manager.get("language_detection")
        )
        self.phrase_detector = PhraseDetector(
            self.config_manager.get("detection_features", "phrase_detection")
        )
        self.context_analyzer = ContextAnalyzer(
            self.config_manager.get("detection_features", "context_analysis")
        )
        self.database_manager = BullyingDatabaseManager(database_path)
        
        # Initialize cache
        cache_config = self.config_manager.get("detection_features", "real_time_features").get("caching", {})
        self.cache_manager = CacheManager(cache_config)
        
        # Load feature configuration
        self.features_config = self.config_manager.get("detection_features")
        
        logger.info("AdvancedBullyingDetector initialized successfully")
    
    async def detect_async(self, text: str) -> AnalysisResult:
        """Asynchronous detection method"""
        # Check cache first
        cache_key = self.cache_manager.create_key(text)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Perform detection
        result = await self._perform_detection(text)
        
        # Cache result
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def detect(self, text: str) -> AnalysisResult:
        """Synchronous detection method"""
        return asyncio.run(self.detect_async(text))
    
    async def _perform_detection(self, text: str) -> AnalysisResult:
        """Perform the actual detection"""
        if not text or not text.strip():
            return AnalysisResult(
                is_bullying=False,
                confidence=0.0,
                severity=Severity.NONE,
                detections=[],
                languages=[],
                metadata={"reason": "empty_text"}
            )
        
        # Normalize text
        normalized_text = self.text_normalizer.normalize(text)
        words = self.text_normalizer.tokenize(normalized_text)
        
        # Detect languages
        languages = self.language_detector.detect_languages(text)
        
        # Analyze context
        context = self.context_analyzer.analyze(text, words)
        
        # Collect detections
        detections = []
        
        # Check each language
        for language in languages:
            lang_data = self.database_manager.get_language_data(language)
            if not lang_data:
                continue
            
            # Check single words
            detections.extend(
                await self._check_single_words(words, language, lang_data)
            )
            
            # Check phrases
            detections.extend(
                await self._check_phrases(words, language, lang_data)
            )
            
            # Check patterns
            detections.extend(
                await self._check_patterns(normalized_text, language, lang_data)
            )
        
        # Add context-based detections
        if context["aggression_score"] > 0.3:
            detections.append(Detection(
                type=DetectionType.CONTEXT,
                value="aggressive_context",
                language="neutral",
                severity=Severity.MEDIUM,
                confidence=context["aggression_score"],
                metadata={"indicators": context["indicators"]}
            ))
        
        # Calculate final result
        result = self._calculate_final_result(detections, languages, context)
        
        return result
    
    async def _check_single_words(self, words: List[str], language: str, lang_data: Dict) -> List[Detection]:
        """Check for single word matches"""
        detections = []
        severity_map = lang_data.get("severity_map", {})
        
        for word in words:
            if word in lang_data["single_words"]:
                severity_str = severity_map.get(word, "medium")
                severity = Severity(severity_str)
                
                detections.append(Detection(
                    type=DetectionType.SINGLE_WORD,
                    value=word,
                    language=language,
                    severity=severity,
                    confidence=self.features_config["severity_calculation"]["base_weights"]["single_word"]
                ))
        
        return detections
    
    async def _check_phrases(self, words: List[str], language: str, lang_data: Dict) -> List[Detection]:
        """Check for phrase matches"""
        detections = []
        
        for phrase_info in lang_data["phrases"]:
            result = self.phrase_detector.find_phrase(words, phrase_info["words"])
            
            if result:
                found, score = result
                if found:
                    detections.append(Detection(
                        type=DetectionType.PHRASE,
                        value=phrase_info["text"],
                        language=language,
                        severity=phrase_info["severity"],
                        confidence=self.features_config["severity_calculation"]["base_weights"]["phrase"] * score
                    ))
        
        return detections
    
    async def _check_patterns(self, text: str, language: str, lang_data: Dict) -> List[Detection]:
        """Check for pattern matches"""
        detections = []
        
        for pattern in lang_data["patterns"]:
            matches = pattern.findall(text)
            for match in matches:
                detections.append(Detection(
                    type=DetectionType.PATTERN,
                    value=str(match),
                    language=language,
                    severity=Severity.MEDIUM,
                    confidence=self.features_config["severity_calculation"]["base_weights"]["pattern"]
                ))
        
        return detections
    
    def _calculate_final_result(self, detections: List[Detection], languages: List[str], context: Dict) -> AnalysisResult:
        """Calculate final detection result"""
        if not detections:
            return AnalysisResult(
                is_bullying=False,
                confidence=0.0,
                severity=Severity.NONE,
                detections=[],
                languages=languages,
                metadata={"context": context}
            )
        
        # Calculate total confidence
        total_confidence = sum(d.confidence for d in detections)
        
        # Apply combination rules
        combo_rules = self.features_config["severity_calculation"]["combination_rules"]
        
        if len(detections) >= 4:
            total_confidence *= combo_rules["multiple_detections_boost"]["4_plus_items"]
        elif len(detections) == 3:
            total_confidence *= combo_rules["multiple_detections_boost"]["3_items"]
        elif len(detections) == 2:
            total_confidence *= combo_rules["multiple_detections_boost"]["2_items"]
        
        # Add context boost
        context_boost = min(
            context["aggression_score"] + context["targeting_score"] + context["command_score"],
            combo_rules["context_boost_max"]
        )
        total_confidence += context_boost
        
        # Normalize confidence
        total_confidence = min(total_confidence, 1.0)
        
        # Determine severity
        max_severity = max(detections, key=lambda d: d.severity.weight).severity
        
        # Check threshold
        severity_config = self.features_config["severity_calculation"]["severity_levels"]
        threshold = severity_config.get(max_severity.value, {}).get("threshold", 0.3)
        
        is_bullying = total_confidence >= threshold
        
        return AnalysisResult(
            is_bullying=is_bullying,
            confidence=total_confidence,
            severity=max_severity if is_bullying else Severity.NONE,
            detections=detections,
            languages=languages,
            metadata={
                "context": context,
                "detection_count": len(detections),
                "unique_languages": len(set(d.language for d in detections))
            }
        )
    
    async def detect_batch(self, texts: List[str]) -> List[AnalysisResult]:
        """Batch detection for multiple texts"""
        tasks = [self.detect_async(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def add_words(self, language: str, words: List[str], word_type: str = "single") -> bool:
        """Add new words to the database"""
        return self.database_manager.add_words(language, words, word_type)
    
    def reload_config(self):
        """Reload all configurations"""
        self.config_manager.reload()
        logger.info("Configuration reloaded")


# Create a singleton instance for easy access
_detector_instance = None

def get_detector(config_dir: str = "config", database_path: str = "data/bullying_words.json") -> AdvancedBullyingDetector:
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AdvancedBullyingDetector(config_dir, database_path)
    return _detector_instance
