#!/usr/bin/env python3
"""
Script to check automaton statistics and demonstrate the enhancements.
"""

import sys
sys.path.append('Backend')
from intelligent_detector import IntelligentBullyingDetector

def check_automaton_stats():
    """Check the phrase automaton statistics."""
    detector = IntelligentBullyingDetector()
    stats = detector.phrase_automaton.get_stats()
    
    print('Phrase Automaton Statistics:')
    print('=' * 40)
    print(f'  Total patterns: {stats["pattern_count"]}')
    print(f'  States created: {stats["state_count"]}')
    print(f'  Built: {stats["built"]}')
    print(f'  Categories: {stats["categories"]}')

    # Check what categories are loaded
    print('\nLoaded categories:')
    for category, data in detector.bullying_data.get('bullying_categories', {}).items():
        keywords = len(data.get('keywords', []))
        phrases = len(data.get('multi_word_phrases', []))
        patterns = len(data.get('patterns', []))
        severity = data.get('severity', 'medium')
        boost = data.get('confidence_boost', 0.3)
        print(f'  {category}: {keywords} keywords, {phrases} phrases, {patterns} regex patterns')
        print(f'    -> severity: {severity}, boost: {boost}')
        
        # Show some example keywords
        if keywords > 0:
            example_words = list(data.get('keywords', []))[:3]
            print(f'    -> examples: {example_words}')
        print()

    # Test the enhanced gap handling on a specific example
    print('Testing Enhanced Gap Handling:')
    print('=' * 40)
    
    test_text = "you are really stupid person"
    result = detector.phrase_automaton.search(test_text)
    
    print(f'Text: "{test_text}"')
    print(f'Matches found: {len(result)}')
    
    for match in result:
        print(f'  Pattern: "{match["pattern"]}"')
        print(f'  Category: {match["category"]}')
        print(f'  Match strength: {match["match_strength"]:.3f}')
        print(f'  Confidence boost: {match["confidence_boost"]:.3f}')
        print(f'  Positions: {match["matched_positions"]}')
        print(f'  Span: {match["span"]} words')
        print(f'  Gaps: {match["gaps"]} words')
        if 'quality_metrics' in match:
            quality = match['quality_metrics']
            print(f'  Quality score: {quality["quality_score"]:.3f}')
            print(f'  Compactness: {quality["compactness"]:.3f}')
            print(f'  Importance: {quality["importance_score"]:.3f}')
        print()

if __name__ == '__main__':
    check_automaton_stats()
