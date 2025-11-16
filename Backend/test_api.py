#!/usr/bin/env python3
"""
Test script for the Cyberbullying Detection API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("-" * 50)

def test_single_detection():
    """Test single text detection"""
    print("Testing single text detection...")
    
    test_cases = [
        "Hello, how are you today?",  # Normal text
        "You are so stupid and ugly!",  # Bullying text
        "Nobody likes you, kill yourself",  # High severity bullying
        "That's annoying but whatever",  # Low severity
        "ur such a l0ser and a fr3ak"  # Leet speak bullying
    ]
    
    for text in test_cases:
        print(f"\nTesting: '{text}'")
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={
                "text": text,
                "include_details": True,
                "confidence_threshold": 0.7
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['data']
            print(f"  Bullying: {result['is_bullying']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Severity: {result['severity']}")
            print(f"  Method: {result['detection_method']}")
        else:
            print(f"  Error: {response.status_code} - {response.text}")
    
    print("-" * 50)

def test_batch_detection():
    """Test batch text detection"""
    print("Testing batch detection...")
    
    texts = [
        "Hello friend!",
        "You are stupid",
        "I hate you so much",
        "Great job on your project!",
        "kys loser"
    ]
    
    response = requests.post(
        f"{BASE_URL}/api/batch-detect",
        json={
            "texts": texts,
            "include_details": False,
            "confidence_threshold": 0.6
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        results = data['data']['results']
        print(f"Processed: {data['data']['total_processed']}")
        print(f"Bullying detected: {data['data']['bullying_detected']}")
        
        for result in results:
            print(f"  [{result['index']}] '{texts[result['index']]}' -> {result['is_bullying']} ({result['confidence']:.2f})")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("-" * 50)

def test_statistics():
    """Test statistics endpoint"""
    print("Testing statistics...")
    response = requests.get(f"{BASE_URL}/api/stats")
    
    if response.status_code == 200:
        data = response.json()
        stats = data['data']
        print(f"Total detections: {stats['total_detections']}")
        print(f"Local detections: {stats['local_detections']}")
        print(f"OpenAI detections: {stats['openai_detections']}")
        print(f"Bullying found: {stats['bullying_found']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("-" * 50)

def test_add_words():
    """Test adding new bullying words"""
    print("Testing add words...")
    
    new_words = ["meanie", "butthead", "dumbface"]
    
    response = requests.post(
        f"{BASE_URL}/api/add-words",
        json={"words": new_words}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Words added: {data['data']['words_added']}")
        print(f"Total words submitted: {data['data']['total_words']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    print("-" * 50)

def main():
    """Run all tests"""
    print("Cyberbullying Detection API Tests")
    print("=" * 50)
    
    try:
        test_health_check()
        time.sleep(1)
        
        test_single_detection()
        time.sleep(1)
        
        test_batch_detection()
        time.sleep(1)
        
        test_add_words()
        time.sleep(1)
        
        test_statistics()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask server is running on http://localhost:5000")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
