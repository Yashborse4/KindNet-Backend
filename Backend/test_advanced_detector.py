"""
Test suite for Advanced Multilingual Bullying Detector
"""

import asyncio
import time
from datetime import datetime
from advanced_detector import get_detector, Severity, DetectionType, AnalysisResult


def print_result(text: str, result: AnalysisResult, expected: bool = None):
    """Pretty print test result"""
    status = "✓" if result.is_bullying == expected else "✗" if expected is not None else "?"
    print(f"\n{status} Text: '{text}'")
    print(f"  Bullying: {result.is_bullying} | Confidence: {result.confidence:.2%}")
    print(f"  Severity: {result.severity.value} | Languages: {result.languages}")
    
    if result.detections:
        print("  Detections:")
        for detection in result.detections:
            print(f"    - {detection.type.value}: '{detection.value}' ({detection.language}) "
                  f"[severity: {detection.severity.value}, confidence: {detection.confidence:.2f}]")
    
    if result.metadata.get("context", {}).get("indicators"):
        print(f"  Context indicators: {result.metadata['context']['indicators']}")


async def test_basic_detection():
    """Test basic detection functionality"""
    print("=" * 80)
    print("BASIC DETECTION TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    test_cases = [
        ("This is a normal message", False),
        ("You are stupid", True),
        ("I hate you so much", True),
        ("nobody likes you", True),
        ("kill yourself", True),
        ("Have a great day!", False),
        ("You're amazing", False)
    ]
    
    for text, expected in test_cases:
        result = await detector.detect_async(text)
        print_result(text, result, expected)


async def test_multilingual_detection():
    """Test multilingual detection"""
    print("\n" + "=" * 80)
    print("MULTILINGUAL DETECTION TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    test_cases = [
        # Hindi tests
        ("chutiya", True),
        ("teri maa ki", True),
        ("tumhara kya haal hai", False),
        
        # Marathi tests
        ("tujhi aai", True),
        ("kay mhanto", False),
        
        # Mixed language
        ("you are a chutiya", True),
        ("stupid gandu", True)
    ]
    
    for text, expected in test_cases:
        result = await detector.detect_async(text)
        print_result(text, result, expected)


async def test_advanced_features():
    """Test advanced features like context detection"""
    print("\n" + "=" * 80)
    print("ADVANCED FEATURES TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    test_cases = [
        # Aggression indicators
        ("YOU ARE STUPID!!!!!!", True),
        ("I HATE YOU SO MUCH!!!!", True),
        
        # Leet speak
        ("y0u 4r3 5tup1d", True),
        ("fuk u", True),
        
        # Elongation
        ("stuuuuupid", True),
        ("you're sooooo dumb", True),
        
        # Context without explicit bad words
        ("GO AWAY!!!! NOBODY WANTS YOU HERE!!!!", True)
    ]
    
    for text, expected in test_cases:
        result = await detector.detect_async(text)
        print_result(text, result, expected)


async def test_phrase_flexibility():
    """Test flexible phrase matching"""
    print("\n" + "=" * 80)
    print("PHRASE FLEXIBILITY TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    test_cases = [
        # Exact match
        ("nobody likes you", True),
        
        # With words in between
        ("nobody really likes you", True),
        ("nobody actually really likes you at all", True),
        
        # Different phrase structures
        ("go and die", True),
        ("you should really kill yourself", True)
    ]
    
    for text, expected in test_cases:
        result = await detector.detect_async(text)
        print_result(text, result, expected)


async def test_performance():
    """Test performance and caching"""
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    # Test single detection speed
    text = "you are stupid and nobody likes you"
    
    # First call (no cache)
    start = time.time()
    result1 = await detector.detect_async(text)
    time1 = (time.time() - start) * 1000
    
    # Second call (cached)
    start = time.time()
    result2 = await detector.detect_async(text)
    time2 = (time.time() - start) * 1000
    
    print(f"First call: {time1:.2f}ms")
    print(f"Cached call: {time2:.2f}ms")
    print(f"Speed improvement: {time1/time2:.1f}x")
    
    # Test batch detection
    texts = [
        "you are stupid",
        "nobody likes you",
        "have a nice day",
        "chutiya gandu",
        "kill yourself"
    ] * 10  # 50 texts
    
    start = time.time()
    results = await detector.detect_batch(texts)
    batch_time = (time.time() - start) * 1000
    
    print(f"\nBatch detection of {len(texts)} texts: {batch_time:.2f}ms")
    print(f"Average per text: {batch_time/len(texts):.2f}ms")
    
    bullying_count = sum(1 for r in results if r.is_bullying)
    print(f"Bullying detected in {bullying_count}/{len(texts)} texts")


async def test_severity_levels():
    """Test severity level detection"""
    print("\n" + "=" * 80)
    print("SEVERITY LEVEL TESTS")
    print("=" * 80)
    
    detector = get_detector()
    
    test_cases = [
        ("you're weird", Severity.LOW),
        ("you're stupid", Severity.MEDIUM),
        ("you're worthless", Severity.MEDIUM),
        ("kill yourself", Severity.HIGH),
        ("I will kill you", Severity.HIGH)
    ]
    
    for text, expected_severity in test_cases:
        result = await detector.detect_async(text)
        severity_match = "✓" if result.severity == expected_severity else "✗"
        print(f"\n{severity_match} Text: '{text}'")
        print(f"  Expected severity: {expected_severity.value}")
        print(f"  Detected severity: {result.severity.value}")
        print(f"  Confidence: {result.confidence:.2%}")


async def test_add_words():
    """Test adding new words dynamically"""
    print("\n" + "=" * 80)
    print("DYNAMIC WORD ADDITION TEST")
    print("=" * 80)
    
    detector = get_detector()
    
    # Test with a new word
    test_word = "testbadword123"
    
    # Should not detect initially
    result1 = await detector.detect_async(test_word)
    print(f"Before adding: '{test_word}' - Bullying: {result1.is_bullying}")
    
    # Add the word
    detector.add_words("english", [test_word], "single")
    
    # Should detect now
    result2 = await detector.detect_async(test_word)
    print(f"After adding: '{test_word}' - Bullying: {result2.is_bullying}")
    
    # Test adding a phrase
    test_phrase = "test bad phrase"
    
    result3 = await detector.detect_async(test_phrase)
    print(f"\nBefore adding phrase: '{test_phrase}' - Bullying: {result3.is_bullying}")
    
    detector.add_words("english", [test_phrase], "phrase")
    
    result4 = await detector.detect_async(test_phrase)
    print(f"After adding phrase: '{test_phrase}' - Bullying: {result4.is_bullying}")


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ADVANCED MULTILINGUAL BULLYING DETECTOR TEST SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run all test suites
    await test_basic_detection()
    await test_multilingual_detection()
    await test_advanced_features()
    await test_phrase_flexibility()
    await test_severity_levels()
    await test_add_words()
    await test_performance()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
