#!/usr/bin/env python3
"""
Test script for enhanced phrase boundary detection with adaptive gap limits.
"""

import sys
sys.path.append('Backend')
from intelligent_detector import IntelligentBullyingDetector

def test_enhanced_phrase_detection():
    """Test the enhanced phrase detection system."""
    # Initialize detector
    detector = IntelligentBullyingDetector()
    
    test_cases = [
        # Test adaptive gap handling
        'you are very stupid and ugly',  # Should match with gap
        'you are really really stupid',  # Should match with repetition gap
        'you stupid person are ugly',    # Should match with word in between
        'kill yourself now',             # Should match threat pattern
        'go and die somewhere',          # Should match with gap
        'I think you are kind of stupid honestly',  # Longer gap test
        
        # Test importance weighting
        'you kill me',                   # High importance words
        'you and your stupid face',      # Multiple target words
        'nobody really likes you much',  # Social exclusion with gaps
        'such a complete loser',         # Pattern with filler words
        
        # Test boundary cases
        'you',                          # Single word (should have low confidence)
        'stupid',                       # Single important word
        'hello how are you today',      # Polite phrase (should not trigger)
        'thank you very much'           # Another polite phrase
    ]
    
    print('Testing Enhanced Phrase Detection with Adaptive Gap Limits:')
    print('=' * 70)
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect_bullying_enhanced(text)
        print(f'Test {i:2d}: "{text}"')
        print(f'         Bullying: {result["is_bullying"]} (confidence: {result["confidence"]:.2f})')
        
        if result['detected_categories']:
            for cat in result['detected_categories']:
                print(f'         Category: {cat["category"]} (score: {cat["score"]:.2f})')
        
        # Show quality metrics if available
        if 'quality_metrics' in result:
            print(f'         Method: {result.get("detection_method", "unknown")}')
        
        print()

if __name__ == '__main__':
    try:
        test_enhanced_phrase_detection()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
