#!/usr/bin/env python3
"""
Test script to verify multi-word phrase detection
"""

from intelligent_detector import IntelligentBullyingDetector
import json

def test_multiword_detection():
    """Test detection of multi-word phrases"""
    detector = IntelligentBullyingDetector()
    
    print("=" * 80)
    print("MULTI-WORD PHRASE DETECTION TEST")
    print("=" * 80)
    
    # Test cases for multi-word phrases
    test_cases = [
        # Exact matches
        ("nobody likes you", True, "Exact phrase match"),
        ("gandi soch", True, "Hindi multi-word phrase"),
        ("teri maa ki", True, "Hindi multi-word phrase"),
        ("raand ka baccha", True, "Hindi multi-word phrase"),
        ("ullu de pathe", True, "Hindi multi-word phrase"),
        
        # Phrases with words in between
        ("nobody really likes you", True, "Phrase with word in between"),
        ("nobody actually likes you at all", True, "Phrase with multiple words in between"),
        ("teri maa ki baat", True, "Hindi phrase with extra word"),
        
        # Individual words from phrases should also be detected
        ("nobody", True, "Individual word from phrase"),
        ("likes", True, "Individual word from phrase"),
        ("gandi", True, "Individual word from Hindi phrase"),
        ("soch", True, "Individual word from Hindi phrase"),
        ("teri", True, "Individual word from Hindi phrase"),
        ("maa", True, "Individual word from Hindi phrase"),
        
        # Partial matches shouldn't trigger full phrase confidence
        ("I like you", False, "Partial phrase - should not trigger"),
        ("nobody", True, "Single word from phrase should still be detected"),
        
        # Mixed phrases
        ("you are stupid and nobody likes you", True, "Multiple bullying elements"),
        ("teri maa ka bhenchod", True, "Multiple Hindi profanities"),
        
        # Safe phrases
        ("I really like you", False, "Safe phrase"),
        ("You are amazing", False, "Positive phrase"),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_bullying, description in test_cases:
        result = detector.detect_bullying_enhanced(text)
        is_bullying = result['is_bullying']
        confidence = result['confidence']
        detected_items = []
        
        # Extract detected items
        for category in result.get('detected_categories', []):
            detected_items.extend(category.get('items', []))
        
        is_correct = is_bullying == expected_bullying
        if is_correct:
            passed += 1
            status = "✓"
        else:
            failed += 1
            status = "✗"
        
        print(f"{status} '{text}' - {description}")
        print(f"  Expected: {expected_bullying}, Got: {is_bullying}")
        print(f"  Confidence: {confidence:.2%}")
        if detected_items:
            print(f"  Detected: {detected_items}")
        if not is_correct:
            print(f"  ⚠️  DETECTION ERROR!")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = passed + failed
    accuracy = (passed / total * 100) if total > 0 else 0
    print(f"Passed: {passed}/{total} ({accuracy:.1f}%)")
    print(f"Failed: {failed}")
    
    # Check preprocessing
    print("\n" + "=" * 80)
    print("PREPROCESSING CHECK")
    print("=" * 80)
    
    # Check how multi-word phrases are stored
    for category, patterns in detector.category_patterns.items():
        if patterns.get('multi_word_phrases'):
            print(f"\n{category}:")
            print(f"  Multi-word phrases: {patterns['multi_word_phrases'][:5]}...")  # Show first 5
            print(f"  Total multi-word phrases: {len(patterns['multi_word_phrases'])}")
            print(f"  Sample keywords: {list(patterns['keywords'])[:10]}...")  # Show first 10

def test_phrase_sequence_detection():
    """Test the phrase sequence detection method"""
    detector = IntelligentBullyingDetector()
    
    print("\n" + "=" * 80)
    print("PHRASE SEQUENCE DETECTION TEST")
    print("=" * 80)
    
    test_cases = [
        ("nobody likes you", "nobody likes you", True),
        ("nobody really likes you", "nobody likes you", True),
        ("nobody actually really likes you", "nobody likes you", True),
        ("you likes nobody", "nobody likes you", False),
        ("teri maa ki gaali", "teri maa ki", True),
        ("teri behen ki maa", "teri maa ki", False),
    ]
    
    for text, phrase, expected in test_cases:
        result = detector._check_phrase_sequence(text, phrase)
        status = "✓" if result == expected else "✗"
        print(f"{status} Text: '{text}' | Phrase: '{phrase}' | Expected: {expected}, Got: {result}")

if __name__ == "__main__":
    test_multiword_detection()
    test_phrase_sequence_detection()
