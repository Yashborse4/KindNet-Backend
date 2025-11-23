#!/usr/bin/env python3
"""
Test script for pattern caching and preprocessing optimization.
"""

import sys
sys.path.append('Backend')
from intelligent_detector import IntelligentBullyingDetector
import time

def test_caching_system():
    """Test the caching and preprocessing optimization system."""
    print('Testing Pattern Caching and Preprocessing Optimization:')
    print('=' * 70)
    
    # Initialize detector
    detector = IntelligentBullyingDetector()
    
    # Get cache statistics
    cache_stats = detector.get_cache_statistics()
    print('Cache Statistics:')
    print(f'  Pattern cache size: {cache_stats["pattern_cache_size"]}')
    print(f'  Normalized text cache size: {cache_stats["normalized_text_cache_size"]}')
    print(f'  Keyword lookup size: {cache_stats["keyword_lookup_size"]}')
    print(f'  Phrase index size: {cache_stats["phrase_index_size"]}')
    print(f'  Total cache entries: {cache_stats["total_cache_entries"]}')
    print()
    
    # Test quick keyword lookup
    print('Testing Quick Keyword Lookup (O(1)):')
    test_words = ['stupid', 'kill', 'hurt', 'nonexistent', 'freak']
    for word in test_words:
        lookup_result = detector._quick_keyword_lookup(word)
        if lookup_result:
            print(f'  "{word}": found -> {lookup_result["category"]} (boost: {lookup_result["confidence_boost"]})')
        else:
            print(f'  "{word}": not found')
    print()
    
    # Test indexed phrase lookup
    print('Testing Indexed Phrase Lookup:')
    test_phrases = [
        ['nobody', 'likes', 'you'],
        ['kill', 'yourself'],
        ['stupid', 'ugly'],
        ['random', 'words']
    ]
    
    for words in test_phrases:
        phrase_results = detector._indexed_phrase_lookup(words)
        phrase_str = ' '.join(words)
        if phrase_results:
            for result in phrase_results:
                print(f'  "{phrase_str}": found -> {result["category"]} (boost: {result["confidence_boost"]})')
        else:
            print(f'  "{phrase_str}": not found in index')
    print()
    
    # Test cached vs non-cached text normalization performance
    print('Testing Text Normalization Caching Performance:')
    test_texts = [
        "you are soooo st*pid!!!",
        "k1ll yourself now",
        "what a l0ser you are",
        "f4ck off and die",
        "you are soooo st*pid!!!"  # Duplicate to test cache hit
    ]
    
    # First run (cache misses)
    start_time = time.time()
    for text in test_texts:
        normalized = detector._cached_text_normalize(text)
    first_run_time = time.time() - start_time
    
    # Second run (cache hits)
    start_time = time.time()
    for text in test_texts:
        normalized = detector._cached_text_normalize(text)
    second_run_time = time.time() - start_time
    
    print(f'  First run (cache misses): {first_run_time:.6f} seconds')
    print(f'  Second run (cache hits): {second_run_time:.6f} seconds')
    print(f'  Speed improvement: {first_run_time/second_run_time:.2f}x faster')
    print()
    
    # Show updated cache statistics
    cache_stats_after = detector.get_cache_statistics()
    print('Updated Cache Statistics:')
    print(f'  Normalized text cache size: {cache_stats_after["normalized_text_cache_size"]}')
    print()
    
    # Test full detection with optimized system
    print('Testing Full Detection with Optimized System:')
    test_cases = [
        'you are really stupid and ugly',
        'kill yourself now please',
        'nobody likes you at all',
        'such a complete freak'
    ]
    
    total_time = 0
    for text in test_cases:
        start_time = time.time()
        result = detector.detect_bullying_enhanced(text)
        detection_time = time.time() - start_time
        total_time += detection_time
        
        print(f'  Text: "{text}"')
        print(f'    Bullying: {result["is_bullying"]} (confidence: {result["confidence"]:.2f})')
        print(f'    Processing time: {detection_time:.6f} seconds')
        if result['detected_categories']:
            for cat in result['detected_categories']:
                print(f'    Category: {cat["category"]} (score: {cat["score"]:.2f})')
        print()
    
    print(f'Total detection time: {total_time:.6f} seconds')
    print(f'Average time per detection: {total_time/len(test_cases):.6f} seconds')

if __name__ == '__main__':
    try:
        test_caching_system()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
