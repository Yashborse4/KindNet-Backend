#!/usr/bin/env python3
"""
Comprehensive test script for multilingual bullying detection
Tests multi-word phrase detection and Hindi/Marathi language support
"""

from multilingual_detector import MultilingualBullyingDetector
import json
from datetime import datetime

def print_result(text, result, expected=None):
    """Pretty print test result"""
    status = "✓" if result['is_bullying'] == expected else "✗" if expected is not None else "?"
    print(f"\n{status} Text: '{text}'")
    print(f"  Bullying: {result['is_bullying']} | Confidence: {result['confidence']:.2%}")
    print(f"  Severity: {result['severity']} | Languages: {result['detected_languages']}")
    
    if result['detected_items']:
        print("  Detected items:")
        for item in result['detected_items']:
            print(f"    - {item['type']}: '{item['value']}' ({item['language']}) "
                  f"[severity: {item['severity']}, confidence: {item['confidence']:.2f}]")

def test_english_phrases():
    """Test English multi-word phrase detection"""
    print("=" * 80)
    print("ENGLISH MULTI-WORD PHRASE TESTS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    test_cases = [
        # Exact phrase matches
        ("nobody likes you", True),
        ("kill yourself", True),
        ("go die", True),
        
        # Phrases with words in between
        ("nobody really likes you", True),
        ("nobody actually really likes you at all", True),
        ("go and die", True),
        ("you should kill yourself", True),
        
        # Partial phrases that shouldn't fully match
        ("I like you", False),
        ("nobody", True),  # Single word should still be detected if in database
        
        # Multiple bullying elements
        ("you are stupid and nobody likes you", True),
        ("kill yourself you worthless idiot", True),
        
        # Safe phrases
        ("I really like you", False),
        ("You are amazing", False),
        ("Have a great day", False),
    ]
    
    for text, expected in test_cases:
        result = detector.detect_bullying(text)
        print_result(text, result, expected)

def test_hindi_detection():
    """Test Hindi language detection with multi-word phrases"""
    print("\n" + "=" * 80)
    print("HINDI LANGUAGE TESTS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    test_cases = [
        # Single Hindi words
        ("chutiya", True),
        ("gandu", True),
        ("madarchod", True),
        ("bhenchod", True),
        
        # Hindi multi-word phrases
        ("teri maa ki", True),
        ("ullu ka patha", True),
        ("gaand maaru", True),
        ("teri maa ka bhosda", True),
        
        # Hindi phrases with words in between
        ("teri pyari maa ki", True),
        ("ullu ka bada patha", True),
        
        # Mixed Hindi-English
        ("you are a chutiya", True),
        ("stupid gandu", True),
        ("teri maa ki, you idiot", True),
        
        # Devanagari script (if supported)
        ("तेरी माँ की", True),
        ("चुतिया", True),
        
        # Safe Hindi phrases
        ("acha hai", False),
        ("kya baat hai", False),
        ("dhanyawad", False),
    ]
    
    for text, expected in test_cases:
        result = detector.detect_bullying(text)
        print_result(text, result, expected)

def test_marathi_detection():
    """Test Marathi language detection"""
    print("\n" + "=" * 80)
    print("MARATHI LANGUAGE TESTS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    test_cases = [
        # Marathi offensive words
        ("bokya", True),
        ("gandya", True),
        ("lavdya", True),
        
        # Marathi multi-word phrases
        ("aai ghalya", True),
        ("tujhi aai", True),
        
        # Safe Marathi phrases
        ("kay mhanto", False),
        ("dhanyawad tumhala", False),
    ]
    
    for text, expected in test_cases:
        result = detector.detect_bullying(text)
        print_result(text, result, expected)

def test_word_combinations():
    """Test detection of word combinations and sequences"""
    print("\n" + "=" * 80)
    print("WORD COMBINATION TESTS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    test_cases = [
        # 2-word combinations
        ("stupid idiot", True),
        ("fat ugly", True),
        ("worthless loser", True),
        
        # 3-word combinations
        ("you stupid idiot", True),
        ("ugly fat loser", True),
        ("go kill yourself", True),
        
        # Combinations with conjunctions
        ("stupid and ugly", True),
        ("fat or ugly", True),
        ("idiot, moron, loser", True),
        
        # Hindi combinations
        ("chutiya gandu", True),
        ("madarchod bhenchod", True),
        ("teri maa ki chut", True),
        
        # Mixed language combinations
        ("stupid chutiya", True),
        ("you gandu idiot", True),
        ("bokya stupid", True),
    ]
    
    for text, expected in test_cases:
        result = detector.detect_bullying(text)
        print_result(text, result, expected)

def test_context_sensitivity():
    """Test context-sensitive detection"""
    print("\n" + "=" * 80)
    print("CONTEXT SENSITIVITY TESTS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    test_cases = [
        # Single words that need context
        ("die", False),  # Alone should not trigger
        ("you should die", True),  # With context should trigger
        ("kill", False),  # Alone should not trigger
        ("kill yourself", True),  # With context should trigger
        
        # Severity based on context
        ("stupid", True),
        ("you are so stupid", True),  # Should have higher confidence
        ("stupid stupid stupid", True),  # Repetition increases severity
        
        # False positives to avoid
        ("I'm studying die casting", False),
        ("kill the process", False),
        ("nobody knows", False),  # 'nobody' alone in different context
    ]
    
    for text, expected in test_cases:
        result = detector.detect_bullying(text)
        print_result(text, result, expected)

def test_add_new_words():
    """Test adding new words to the database"""
    print("\n" + "=" * 80)
    print("ADD NEW WORDS TEST")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    # Test before adding
    result_before = detector.detect_bullying("bakchod")
    print(f"Before adding - 'bakchod': {result_before['is_bullying']}")
    
    # Add new Hindi words
    added = detector.add_words(['bakchod', 'haraamkhor'], language='hindi', word_type='single')
    print(f"\nAdded {added} new Hindi words")
    
    # Test after adding
    result_after = detector.detect_bullying("bakchod")
    print(f"After adding - 'bakchod': {result_after['is_bullying']}")
    
    # Add new phrase
    added_phrase = detector.add_words(['tera baap ka'], language='hindi', word_type='phrase')
    print(f"\nAdded {added_phrase} new Hindi phrase")
    
    # Test phrase
    result_phrase = detector.detect_bullying("tera baap ka")
    print(f"After adding - 'tera baap ka': {result_phrase['is_bullying']}")

def run_performance_test():
    """Test detection performance"""
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    
    # Test with various text lengths
    test_texts = [
        "short text",
        "this is a medium length text with some words",
        "this is a much longer text that contains multiple sentences and should test how well the detector performs with longer content including some bad words like stupid and idiot",
        "मैं तुम्हें बहुत पसंद करता हूं लेकिन तुम एक chutiya हो और nobody likes you because you are worthless"
    ]
    
    import time
    for text in test_texts:
        start = time.time()
        result = detector.detect_bullying(text)
        end = time.time()
        print(f"\nText length: {len(text)} chars, {len(text.split())} words")
        print(f"Detection time: {(end - start) * 1000:.2f}ms")
        print(f"Bullying detected: {result['is_bullying']}")

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("MULTILINGUAL BULLYING DETECTION TEST SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run all test categories
    test_english_phrases()
    test_hindi_detection()
    test_marathi_detection()
    test_word_combinations()
    test_context_sensitivity()
    test_add_new_words()
    run_performance_test()
    
    # Print statistics
    print("\n" + "=" * 80)
    print("TEST STATISTICS")
    print("=" * 80)
    
    detector = MultilingualBullyingDetector()
    stats = detector.get_statistics()
    print(f"Total detections: {stats['total_detections']}")
    print(f"Bullying detected: {stats['bullying_detected']}")
    print(f"Word matches: {stats['word_matches']}")
    print(f"Phrase matches: {stats['phrase_matches']}")
    print(f"Languages detected: {dict(stats['languages_detected'])}")

if __name__ == "__main__":
    main()
