#!/usr/bin/env python3
"""
Comprehensive test to identify word detection issues
"""

from intelligent_detector import IntelligentBullyingDetector
import json

def test_word_detection():
    """Test word detection capabilities"""
    detector = IntelligentBullyingDetector()
    
    # Test cases with expected results
    test_cases = [
        # Basic insults that should be detected
        ("stupid", True, "Basic insult"),
        ("idiot", True, "Basic insult"),
        ("loser", True, "Basic insult"),
        
        # Hindi profanity that should be detected
        ("madarchod", True, "Hindi profanity"),
        ("bhenchod", True, "Hindi profanity"),
        ("chutiya", True, "Hindi profanity"),
        ("gandu", True, "Hindi profanity"),
        
        # Variations that should be normalized and detected
        ("maderchood", True, "Variation of madarchod"),
        ("maderchod", True, "Variation of madarchod"),
        ("benchod", True, "Variation of bhenchod"),
        ("benchood", True, "Variation of bhenchod"),
        
        # Leet speak that should be detected
        ("stup1d", True, "Leet speak version of stupid"),
        ("1d10t", True, "Leet speak version of idiot"),
        ("b1tch", True, "Leet speak version of bitch"),
        
        # Phrases that should be detected
        ("kill yourself", True, "Threat phrase"),
        ("nobody likes you", True, "Social exclusion phrase"),
        ("you are stupid", True, "Insult phrase"),
        ("go die", True, "Threat phrase"),
        
        # Mixed case that should be detected
        ("STUPID", True, "Uppercase insult"),
        ("StUpId", True, "Mixed case insult"),
        ("MaDaRcHoD", True, "Mixed case profanity"),
        
        # Context-based detection
        ("You are so worthless", True, "Context-based insult"),
        ("I hate you", True, "Hate speech"),
        
        # Safe words that should NOT be detected
        ("hello", False, "Safe greeting"),
        ("thank you", False, "Polite phrase"),
        ("good morning", False, "Safe greeting"),
        ("how are you", False, "Normal question"),
        ("nice to meet you", False, "Polite phrase"),
        
        # Edge cases
        ("", False, "Empty string"),
        ("   ", False, "Whitespace only"),
        ("123", False, "Numbers only"),
        ("!!!", False, "Punctuation only"),
        
        # Words that might be in context but not offensive alone
        ("mad", False, "Could be 'mad' as in angry, not offensive"),
        ("die", False, "Could be in non-threatening context"),
    ]
    
    print("=" * 80)
    print("WORD DETECTION TEST RESULTS")
    print("=" * 80)
    
    correct_detections = 0
    false_positives = []
    false_negatives = []
    
    for text, expected_bullying, description in test_cases:
        result = detector.detect_bullying_enhanced(text)
        is_bullying = result['is_bullying']
        confidence = result['confidence']
        severity = result['severity']
        categories = result.get('detected_categories', [])
        
        # Check if detection matches expectation
        is_correct = is_bullying == expected_bullying
        correct_detections += 1 if is_correct else 0
        
        # Track errors
        if is_bullying and not expected_bullying:
            false_positives.append((text, description, confidence))
        elif not is_bullying and expected_bullying:
            false_negatives.append((text, description, confidence))
        
        # Print result
        status = "✓" if is_correct else "✗"
        print(f"{status} '{text}' - {description}")
        print(f"  Expected: {expected_bullying}, Got: {is_bullying}")
        print(f"  Confidence: {confidence:.2%}, Severity: {severity}")
        if categories:
            cat_names = [cat['category'] for cat in categories]
            print(f"  Categories: {', '.join(cat_names)}")
        if not is_correct:
            print(f"  ⚠ DETECTION ERROR!")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    accuracy = (correct_detections / len(test_cases)) * 100
    print(f"Accuracy: {correct_detections}/{len(test_cases)} ({accuracy:.1f}%)")
    
    if false_positives:
        print(f"\nFalse Positives ({len(false_positives)}):")
        for text, desc, conf in false_positives:
            print(f"  - '{text}' ({desc}) - Confidence: {conf:.2%}")
    
    if false_negatives:
        print(f"\nFalse Negatives ({len(false_negatives)}):")
        for text, desc, conf in false_negatives:
            print(f"  - '{text}' ({desc}) - Confidence: {conf:.2%}")
    
    # Check normalization
    print("\n" + "=" * 80)
    print("NORMALIZATION TEST")
    print("=" * 80)
    
    normalization_tests = [
        ("madarchod", "madarchod", "Original"),
        ("maderchood", "madarchod", "Variant 1"),
        ("maderchod", "madarchod", "Variant 2"),
        ("motherchod", "madarchod", "English variant"),
        ("m@d@rch0d", "madarchod", "Leet speak"),
        ("MADARCHOD", "madarchod", "Uppercase"),
        ("MaDaRcHoD", "madarchod", "Mixed case"),
    ]
    
    for original, expected_norm, desc in normalization_tests:
        normalized = detector._normalize_text(original)
        print(f"'{original}' -> '{normalized}' (Expected: '{expected_norm}') - {desc}")
        if expected_norm.lower() in normalized.lower():
            print("  ✓ Normalization working")
        else:
            print("  ✗ Normalization issue!")
    
    # Check database content
    print("\n" + "=" * 80)
    print("DATABASE CHECK")
    print("=" * 80)
    
    # Check if key words are in database
    important_words = ["madarchod", "bhenchod", "stupid", "idiot", "chutiya", "gandu"]
    
    for word in important_words:
        found = False
        for category, data in detector.category_patterns.items():
            if word in data['keywords']:
                found = True
                print(f"✓ '{word}' found in category: {category}")
                break
        if not found:
            print(f"✗ '{word}' NOT found in database!")
    
    return accuracy

if __name__ == "__main__":
    accuracy = test_word_detection()
    print(f"\nFinal Accuracy: {accuracy:.1f}%")
