#!/usr/bin/env python3
"""
Comprehensive test to identify API detection issues
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_specific_words():
    """Test specific words that should be detected"""
    print("TESTING SPECIFIC WORD DETECTION VIA API")
    print("=" * 80)
    
    test_words = [
        # Basic insults
        ("stupid", True, "Basic insult"),
        ("madarchod", True, "Hindi profanity"),
        ("bhenchod", True, "Hindi profanity"),
        ("You are stupid", True, "Insult phrase"),
        ("You're so stupid and ugly", True, "Multiple insults"),
        
        # Mixed case
        ("STUPID", True, "Uppercase"),
        ("StUpId", True, "Mixed case"),
        ("MaDaRcHoD", True, "Mixed case profanity"),
        
        # Leet speak
        ("stup1d", True, "Leet speak"),
        ("b1tch", True, "Leet speak"),
        ("ur such a l0ser", True, "Leet speak phrase"),
        
        # Phrases
        ("kill yourself", True, "Threat"),
        ("nobody likes you", True, "Social exclusion"),
        ("go die", True, "Threat"),
        ("I hate you", True, "Hate speech"),
        
        # Context required
        ("die", False, "Single word without context"),
        ("You should die", True, "Die with threatening context"),
        
        # Safe words
        ("hello", False, "Greeting"),
        ("thank you", False, "Polite"),
    ]
    
    results = []
    for text, expected, description in test_words:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={"text": text}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['data']
            is_correct = result['is_bullying'] == expected
            
            print(f"{'✓' if is_correct else '✗'} '{text}' - {description}")
            print(f"  Expected: {expected}, Got: {result['is_bullying']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            if 'detected_categories' in result:
                categories = [cat.get('category', 'unknown') for cat in result.get('detected_categories', [])]
                if categories:
                    print(f"  Categories: {', '.join(categories)}")
            if 'openai_reasoning' in result and result['openai_reasoning']:
                print(f"  OpenAI: {result['openai_reasoning'][:100]}...")
            if not is_correct:
                print(f"  ⚠️  DETECTION ERROR!")
            print()
            
            results.append({
                'text': text,
                'expected': expected,
                'actual': result['is_bullying'],
                'confidence': result['confidence'],
                'correct': is_correct
            })
        else:
            print(f"Error testing '{text}': {response.status_code}")
    
    # Summary
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("=" * 80)
    print(f"API Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Show failures
    failures = [r for r in results if not r['correct']]
    if failures:
        print("\nFailed detections:")
        for fail in failures:
            print(f"  - '{fail['text']}' (Expected: {fail['expected']}, Got: {fail['actual']}, Conf: {fail['confidence']:.2%})")
    
    return results

def test_context_and_phrases():
    """Test context-aware detection"""
    print("\n\nTESTING CONTEXT AND PHRASE DETECTION")
    print("=" * 80)
    
    test_cases = [
        # Context matters
        ("die", False, "Single word without context"),
        ("go die", True, "Die in threatening context"),
        ("kill", False, "Single word without context"),
        ("kill yourself", True, "Clear threat"),
        ("I will kill you", True, "Direct threat"),
        
        # Phrases
        ("You are so stupid", True, "Clear insult phrase"),
        ("nobody likes you", True, "Social exclusion"),
        ("everyone hates you", True, "Social exclusion"),
        ("you're worthless", True, "Direct insult"),
        
        # Mixed with safe content
        ("Hello stupid", True, "Insult mixed with greeting"),
        ("Thanks idiot", True, "Insult mixed with thanks"),
        
        # Edge cases
        ("ur dumb", True, "Shorthand insult"),
        ("u r stupid", True, "Spaced out insult"),
        ("you    are    stupid", True, "Extra spaces"),
    ]
    
    for text, expected, description in test_cases:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={"text": text, "confidence_threshold": 0.3}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['data']
            is_correct = result['is_bullying'] == expected
            
            print(f"{'✓' if is_correct else '✗'} '{text}' - {description}")
            print(f"  Expected: {expected}, Got: {result['is_bullying']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Method: {result.get('detection_method', 'unknown')}")
            if not is_correct:
                print(f"  ⚠️  DETECTION ERROR!")
                # Debug info
                if 'sentiment_analysis' in result:
                    sentiment = result['sentiment_analysis']
                    print(f"  Sentiment: compound={sentiment.get('compound', 0):.2f}")
                if 'context_analysis' in result:
                    context = result['context_analysis']
                    print(f"  Context score: {context.get('score', 0):.2f}")
            print()

def test_confidence_thresholds():
    """Test different confidence thresholds"""
    print("\n\nTESTING CONFIDENCE THRESHOLDS")
    print("=" * 80)
    
    test_text = "You are stupid"
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Testing text: '{test_text}'")
    for threshold in thresholds:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={"text": test_text, "confidence_threshold": threshold}
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['data']
            print(f"  Threshold {threshold}: Bullying={result['is_bullying']}, Confidence={result['confidence']:.2%}, Method={result.get('detection_method', 'unknown')}")

def main():
    """Run all comprehensive tests"""
    print("COMPREHENSIVE API DETECTION TESTS")
    print("=" * 80)
    
    try:
        # Test specific words
        test_specific_words()
        
        # Test context and phrases
        test_context_and_phrases()
        
        # Test confidence thresholds
        test_confidence_thresholds()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask server is running on http://localhost:5000")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
