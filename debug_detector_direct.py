#!/usr/bin/env python3
"""
Debug the detector behavior to understand API issues
"""

from intelligent_detector import IntelligentBullyingDetector
import json

def test_detector_behavior():
    """Test detector behavior with different scenarios"""
    detector = IntelligentBullyingDetector()
    
    print("TESTING DETECTOR BEHAVIOR")
    print("=" * 80)
    
    # Test cases that are failing in API
    test_cases = [
        ("madarchod", 0.7),
        ("madarchod", 0.3),
        ("bhenchod", 0.7),
        ("stup1d", 0.7),
        ("b1tch", 0.7),
        ("die", 0.7),
        ("You are stupid", 0.7),
        ("You are stupid", 0.3),
    ]
    
    for text, threshold in test_cases:
        print(f"\nTesting: '{text}' with threshold={threshold}")
        result = detector.detect_bullying_enhanced(text, confidence_threshold=threshold)
        
        print(f"  Is Bullying: {result['is_bullying']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Detection Method: {result.get('detection_method', 'unknown')}")
        print(f"  Severity: {result.get('severity', 'none')}")
        
        if 'detected_categories' in result:
            categories = [cat['category'] for cat in result.get('detected_categories', [])]
            if categories:
                print(f"  Categories: {', '.join(categories)}")
        
        if 'openai_reasoning' in result and result['openai_reasoning']:
            print(f"  OpenAI Reasoning: {result['openai_reasoning'][:100]}...")
        
        # Check local detection details
        if 'sentiment_analysis' in result:
            sentiment = result['sentiment_analysis']
            print(f"  Sentiment: compound={sentiment.get('compound', 0):.2f}")
        
        if 'context_analysis' in result:
            context = result['context_analysis']
            print(f"  Context Score: {context.get('score', 0):.2f}")

def check_openai_config():
    """Check if OpenAI is configured"""
    import os
    api_key = os.getenv('OPENAI_API_KEY')
    print("\n" + "=" * 80)
    print("OPENAI CONFIGURATION CHECK")
    print("=" * 80)
    print(f"OpenAI API Key configured: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key length: {len(api_key)} characters")
        print(f"API Key prefix: {api_key[:10]}...")

def test_normalization():
    """Test text normalization"""
    detector = IntelligentBullyingDetector()
    
    print("\n" + "=" * 80)
    print("TEXT NORMALIZATION TEST")
    print("=" * 80)
    
    test_texts = [
        "madarchod",
        "MaDaRcHoD",
        "m@d@rch0d",
        "stup1d",
        "b1tch",
    ]
    
    for text in test_texts:
        normalized = detector._normalize_text(text)
        print(f"'{text}' -> '{normalized}'")

def test_local_check_details():
    """Test the local check method in detail"""
    detector = IntelligentBullyingDetector()
    
    print("\n" + "=" * 80)
    print("LOCAL CHECK DETAILS")
    print("=" * 80)
    
    test_text = "madarchod"
    
    # Get sentiment, context, and intent
    sentiment = detector._analyze_sentiment(test_text)
    context = detector._analyze_context(test_text)
    intent = detector._classify_intent(test_text)
    
    print(f"Testing: '{test_text}'")
    print(f"Sentiment: {sentiment}")
    print(f"Context: {context}")
    print(f"Intent: {intent}")
    
    # Test local check
    local_result = detector._enhanced_local_check(test_text, sentiment, context, intent)
    print(f"\nLocal Check Result:")
    print(f"  Is Bullying: {local_result['is_bullying']}")
    print(f"  Confidence: {local_result['confidence']:.2%}")
    print(f"  Categories: {[cat['category'] for cat in local_result.get('detected_categories', [])]}")

if __name__ == "__main__":
    check_openai_config()
    test_normalization()
    test_local_check_details()
    test_detector_behavior()
