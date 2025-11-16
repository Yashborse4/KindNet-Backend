#!/usr/bin/env python3
"""
Test specific words for bullying detection
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_word(word):
    """Test a single word"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={"text": word, "include_details": True}
        )
        
        if response.status_code == 200:
            data = response.json()['data']
            print(f"'{word}': Bullying={data['is_bullying']}, Confidence={data['confidence']:.2f}, Severity={data['severity']}")
            if data.get('detected_categories'):
                print(f"  Categories: {[cat['category'] for cat in data['detected_categories']]}")
            if data.get('explanation'):
                print(f"  Explanation: {data['explanation']}")
        else:
            print(f"'{word}': ERROR - {response.status_code}")
    except Exception as e:
        print(f"'{word}': EXCEPTION - {str(e)}")

def main():
    """Test various bullying words"""
    test_words = [
        # Basic English words
        "stupid", "idiot", "loser", "ugly", "pathetic", "worthless",
        
        # Hindi profanity (original and variants)
        "madarchod", "maderchood", "maderchod", "bhenchod", "benchod", "benchood",
        "randi", "gandu", "chutiya", "bhosdike", "harami", "kutta",
        
        # Normalized test
        "nalak", "dhakkan", "kaamchor",
        
        # Phrases
        "kill yourself", "nobody likes you", "you are stupid",
        
        # Safe words
        "hello", "thank you", "good morning"
    ]
    
    print("Bullying Detection Test Results")
    print("=" * 50)
    
    for word in test_words:
        test_word(word)
        print()

if __name__ == "__main__":
    main()
