import json
import re
import os
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import openai
from openai import OpenAI
from collections import defaultdict
import string

logger = logging.getLogger(__name__)

class BullyingDetector:
    """
    Multi-tier cyberbullying detection system:
    1. Local database check (offline)
    2. OpenAI API fallback (online)
    """
    
    def __init__(self, data_file_path: str = None, openai_api_key: str = None):
        """Initialize the bullying detector"""
        self.data_file_path = data_file_path or os.path.join('data', 'bullying_words.json')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'local_detections': 0,
            'openai_detections': 0,
            'bullying_found': 0,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        # Load local database
        self.bullying_data = self._load_bullying_database()
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Preprocess data for faster lookup
        self._preprocess_data()
        
        logger.info("BullyingDetector initialized successfully")
    
    def _load_bullying_database(self) -> Dict:
        """Load bullying words and phrases from JSON file"""
        try:
            if not os.path.exists(self.data_file_path):
                logger.warning(f"Bullying database file not found: {self.data_file_path}")
                return self._get_default_data()
            
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update metadata
            if 'metadata' in data:
                data['metadata']['total_words'] = len(data.get('bullying_words', []))
                data['metadata']['total_phrases'] = len(data.get('bullying_phrases', []))
            
            logger.info(f"Loaded bullying database with {len(data.get('bullying_words', []))} words and {len(data.get('bullying_phrases', []))} phrases")
            return data
            
        except Exception as e:
            logger.error(f"Error loading bullying database: {str(e)}")
            return self._get_default_data()
    
    def _get_default_data(self) -> Dict:
        """Return minimal default data if file loading fails"""
        return {
            'bullying_words': ['stupid', 'idiot', 'loser', 'ugly', 'fat', 'worthless'],
            'bullying_phrases': ['kill yourself', 'nobody likes you', 'you suck'],
            'severity_levels': {
                'low': ['annoying', 'weird'],
                'medium': ['stupid', 'ugly'],
                'high': ['kill yourself', 'worthless']
            },
            'contextual_patterns': [],
            'intent_indicators': []
        }
    
    def _init_openai_client(self):
        """Initialize OpenAI client if API key is available"""
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                self.openai_client = None
        else:
            logger.warning("OpenAI API key not provided - online detection disabled")
            self.openai_client = None
    
    def _preprocess_data(self):
        """Preprocess data for faster lookup"""
        # Create sets for O(1) lookup
        self.bullying_words_set = set(word.lower() for word in self.bullying_data.get('bullying_words', []))
        self.bullying_phrases_set = set(phrase.lower() for phrase in self.bullying_data.get('bullying_phrases', []))
        
        # Compile regex patterns
        self.compiled_patterns = []
        for pattern_data in self.bullying_data.get('contextual_patterns', []):
            try:
                compiled = re.compile(pattern_data['pattern'], re.IGNORECASE)
                self.compiled_patterns.append({
                    'pattern': compiled,
                    'severity': pattern_data['severity'],
                    'description': pattern_data.get('description', '')
                })
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern_data['pattern']} - {str(e)}")
        
        # Create severity mapping
        self.severity_map = {}
        for severity, words in self.bullying_data.get('severity_levels', {}).items():
            for word in words:
                self.severity_map[word.lower()] = severity
        
        logger.info("Data preprocessing completed")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Handle common character substitutions (leet speak)
        substitutions = {
            '4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's',
            '7': 't', '@': 'a', '$': 's', '!': 'i'
        }
        
        for char, replacement in substitutions.items():
            text = text.replace(char, replacement)
        
        # Remove excessive punctuation but keep structure
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text, handling punctuation"""
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator)
        words = clean_text.split()
        return [word.lower() for word in words if word]
    
    def _check_local_database(self, text: str) -> Dict:
        """Check text against local bullying database"""
        normalized_text = self._normalize_text(text)
        words = self._extract_words(normalized_text)
        
        found_words = []
        found_phrases = []
        found_patterns = []
        max_severity = 'none'
        confidence_score = 0.0
        
        # Check individual words
        for word in words:
            if word in self.bullying_words_set:
                found_words.append(word)
                severity = self.severity_map.get(word, 'medium')
                if self._severity_level(severity) > self._severity_level(max_severity):
                    max_severity = severity
                confidence_score += 0.3
        
        # Check phrases
        for phrase in self.bullying_phrases_set:
            if phrase in normalized_text:
                found_phrases.append(phrase)
                severity = self.severity_map.get(phrase, 'high')
                if self._severity_level(severity) > self._severity_level(max_severity):
                    max_severity = severity
                confidence_score += 0.5
        
        # Check contextual patterns
        for pattern_data in self.compiled_patterns:
            matches = pattern_data['pattern'].findall(normalized_text)
            if matches:
                found_patterns.append({
                    'matches': matches,
                    'severity': pattern_data['severity'],
                    'description': pattern_data['description']
                })
                if self._severity_level(pattern_data['severity']) > self._severity_level(max_severity):
                    max_severity = pattern_data['severity']
                confidence_score += 0.4
        
        # Check intent indicators
        intent_score = self._check_intent_indicators(normalized_text, words)
        confidence_score += intent_score
        
        # Normalize confidence score
        confidence_score = min(confidence_score, 1.0)
        
        # Determine if bullying detected
        is_bullying = (
            len(found_words) > 0 or 
            len(found_phrases) > 0 or 
            len(found_patterns) > 0 or
            confidence_score >= 0.3
        )
        
        return {
            'is_bullying': is_bullying,
            'confidence': confidence_score,
            'severity': max_severity if is_bullying else 'none',
            'detection_method': 'local_database',
            'details': {
                'found_words': found_words,
                'found_phrases': found_phrases,
                'found_patterns': found_patterns,
                'word_count': len(words),
                'total_indicators': len(found_words) + len(found_phrases) + len(found_patterns)
            }
        }
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity string to numeric level"""
        severity_levels = {
            'none': 0, 'low': 1, 'medium': 2, 'high': 3
        }
        return severity_levels.get(severity, 0)
    
    def _check_intent_indicators(self, text: str, words: List[str]) -> float:
        """Check for intent-based indicators"""
        score = 0.0
        
        for indicator in self.bullying_data.get('intent_indicators', []):
            category_score = 0.0
            found_keywords = []
            
            for keyword in indicator.get('keywords', []):
                if isinstance(keyword, str):
                    if keyword.lower() in words or keyword.lower() in text:
                        found_keywords.append(keyword)
                        category_score += 0.1
            
            if found_keywords:
                severity_multiplier = {
                    'low': 0.5, 'medium': 1.0, 'high': 1.5
                }.get(indicator.get('severity', 'medium'), 1.0)
                
                score += category_score * severity_multiplier
        
        return min(score, 0.5)  # Cap intent score contribution
    
    async def _check_openai(self, text: str, local_confidence: float) -> Dict:
        """Use OpenAI API for advanced bullying detection"""
        if not self.openai_client:
            return {
                'is_bullying': False,
                'confidence': 0.0,
                'severity': 'none',
                'detection_method': 'openai_unavailable',
                'details': {'error': 'OpenAI client not available'}
            }
        
        try:
            prompt = f"""
            Analyze the following text for cyberbullying content. Consider context, intent, and potential harm.
            
            Text: "{text}"
            
            Local detection confidence: {local_confidence:.2f}
            
            Please respond with a JSON object containing:
            - is_bullying: boolean
            - confidence: float (0.0 to 1.0)
            - severity: string ("none", "low", "medium", "high")
            - reasoning: string explaining the decision
            - categories: array of detected bullying categories (if any)
            
            Consider these aspects:
            1. Direct insults or name-calling
            2. Threats or encouragement of self-harm
            3. Social exclusion or isolation
            4. Harassment or intimidation
            5. Context and intent behind the message
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in cyberbullying detection. Analyze text objectively and provide accurate assessments."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # Parse OpenAI response
            response_text = response.choices[0].message.content.strip()
            
            try:
                # Try to parse as JSON
                openai_result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("OpenAI returned non-JSON response, attempting fallback parsing")
                openai_result = self._parse_openai_fallback(response_text)
            
            return {
                'is_bullying': openai_result.get('is_bullying', False),
                'confidence': float(openai_result.get('confidence', 0.0)),
                'severity': openai_result.get('severity', 'none'),
                'detection_method': 'openai',
                'details': {
                    'reasoning': openai_result.get('reasoning', ''),
                    'categories': openai_result.get('categories', []),
                    'raw_response': response_text
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                'is_bullying': False,
                'confidence': 0.0,
                'severity': 'none',
                'detection_method': 'openai_error',
                'details': {'error': str(e)}
            }
    
    def _parse_openai_fallback(self, response_text: str) -> Dict:
        """Fallback parsing for non-JSON OpenAI responses"""
        result = {
            'is_bullying': False,
            'confidence': 0.0,
            'severity': 'none',
            'reasoning': response_text,
            'categories': []
        }
        
        # Simple keyword-based parsing
        response_lower = response_text.lower()
        
        if any(word in response_lower for word in ['yes', 'bullying', 'harassment', 'threat']):
            result['is_bullying'] = True
            result['confidence'] = 0.6
            
        if any(word in response_lower for word in ['high', 'severe', 'serious']):
            result['severity'] = 'high'
            result['confidence'] = 0.8
        elif any(word in response_lower for word in ['medium', 'moderate']):
            result['severity'] = 'medium'
            result['confidence'] = 0.6
        elif any(word in response_lower for word in ['low', 'mild']):
            result['severity'] = 'low'
            result['confidence'] = 0.4
        
        return result
    
    def detect_bullying(self, text: str, confidence_threshold: float = 0.7, include_details: bool = True) -> Dict:
        """
        Main detection method with multi-tier approach:
        1. Check local database first
        2. Use OpenAI if local confidence is low or uncertain
        """
        self.stats['total_detections'] += 1
        
        if not text or not text.strip():
            return {
                'is_bullying': False,
                'confidence': 0.0,
                'severity': 'none',
                'detection_method': 'empty_text',
                'details': {} if include_details else None
            }
        
        # Step 1: Check local database
        local_result = self._check_local_database(text.strip())
        self.stats['local_detections'] += 1
        
        # If local detection is confident, use it
        if local_result['confidence'] >= confidence_threshold:
            if local_result['is_bullying']:
                self.stats['bullying_found'] += 1
            
            result = local_result.copy()
            if not include_details:
                result.pop('details', None)
            return result
        
        # Step 2: Use OpenAI for uncertain cases
        try:
            import asyncio
            openai_result = asyncio.run(self._check_openai(text.strip(), local_result['confidence']))
            self.stats['openai_detections'] += 1
            
            # Combine local and OpenAI results
            final_result = self._combine_results(local_result, openai_result, confidence_threshold)
            
            if final_result['is_bullying']:
                self.stats['bullying_found'] += 1
            
            if not include_details:
                final_result.pop('details', None)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in OpenAI detection: {str(e)}")
            # Fallback to local result
            if local_result['is_bullying']:
                self.stats['bullying_found'] += 1
            
            result = local_result.copy()
            if not include_details:
                result.pop('details', None)
            return result
    
    def _combine_results(self, local_result: Dict, openai_result: Dict, threshold: float) -> Dict:
        """Combine local and OpenAI detection results"""
        # Weight the results (local: 40%, OpenAI: 60%)
        local_weight = 0.4
        openai_weight = 0.6
        
        combined_confidence = (
            local_result['confidence'] * local_weight + 
            openai_result['confidence'] * openai_weight
        )
        
        # Determine final severity
        local_severity_level = self._severity_level(local_result['severity'])
        openai_severity_level = self._severity_level(openai_result['severity'])
        
        if openai_severity_level > local_severity_level:
            final_severity = openai_result['severity']
        else:
            final_severity = local_result['severity']
        
        # Final bullying determination
        is_bullying = (
            combined_confidence >= threshold or
            local_result['is_bullying'] or
            openai_result['is_bullying']
        )
        
        return {
            'is_bullying': is_bullying,
            'confidence': combined_confidence,
            'severity': final_severity if is_bullying else 'none',
            'detection_method': 'combined',
            'details': {
                'local_result': local_result,
                'openai_result': openai_result,
                'combination_weights': {
                    'local': local_weight,
                    'openai': openai_weight
                }
            }
        }
    
    def add_bullying_words(self, words: List[str]) -> int:
        """Add new words to the bullying database"""
        added_count = 0
        current_words = set(self.bullying_data.get('bullying_words', []))
        
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in current_words:
                self.bullying_data['bullying_words'].append(word_clean)
                current_words.add(word_clean)
                self.bullying_words_set.add(word_clean)
                added_count += 1
        
        # Save updated database
        if added_count > 0:
            try:
                self._save_database()
                logger.info(f"Added {added_count} new bullying words")
            except Exception as e:
                logger.error(f"Error saving updated database: {str(e)}")
        
        return added_count
    
    def _save_database(self):
        """Save the updated bullying database"""
        # Update metadata
        self.bullying_data['metadata']['total_words'] = len(self.bullying_data.get('bullying_words', []))
        self.bullying_data['metadata']['total_phrases'] = len(self.bullying_data.get('bullying_phrases', []))
        self.bullying_data['metadata']['last_updated'] = datetime.utcnow().isoformat()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
        
        with open(self.data_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.bullying_data, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        self.stats['last_updated'] = datetime.utcnow().isoformat()
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.stats = {
            'total_detections': 0,
            'local_detections': 0,
            'openai_detections': 0,
            'bullying_found': 0,
            'last_updated': datetime.utcnow().isoformat()
        }
        logger.info("Detection statistics reset")
