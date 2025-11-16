#!/usr/bin/env python3
"""
Debug the detector to see what words are loaded
"""

from intelligent_detector import IntelligentBullyingDetector

def main():
    detector = IntelligentBullyingDetector()
    
    print("=== DETECTOR DEBUG INFO ===")
    print(f"Data file path: {detector.data_file_path}")
    print()
    
    print("=== LOADED CATEGORIES ===")
    for category, patterns in detector.category_patterns.items():
        print(f"Category: {category}")
        print(f"  Keywords ({len(patterns['keywords'])}): {list(patterns['keywords'])[:10]}...")
        print(f"  Confidence boost: {patterns['confidence_boost']}")
        print(f"  Severity: {patterns['severity']}")
        print()
    
    # Test specific word processing
    test_words = ['stupid', 'madarchod', 'maderchood']
    
    for word in test_words:
        print(f"=== TESTING WORD: {word} ===")
        normalized = detector._normalize_text(word)
        extracted = detector._extract_words(word.lower())
        
        print(f"  Original: {word}")
        print(f"  Normalized: {normalized}")
        print(f"  Extracted words: {extracted}")
        
        # Check if any category matches
        for category, patterns in detector.category_patterns.items():
            matches = []
            for extracted_word in extracted:
                if extracted_word in patterns['keywords']:
                    matches.append(extracted_word)
            if matches:
                print(f"  Found in category {category}: {matches}")
        print()

if __name__ == "__main__":
    main()
