#!/usr/bin/env python3
"""
Example client for the Cyberbullying Detection API
Shows how to integrate the API into your applications
"""

import requests
import json
from typing import Dict, List, Optional

class BullyingDetectionClient:
    """Client class for the Cyberbullying Detection API"""
    
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        self.api_base_url = api_base_url.rstrip('/')
    
    def detect_bullying(self, text: str, confidence_threshold: float = 0.7, include_details: bool = True) -> Dict:
        """
        Detect bullying in a single text
        
        Args:
            text: The text to analyze
            confidence_threshold: Minimum confidence required (0.0 to 1.0)
            include_details: Whether to include detailed detection information
            
        Returns:
            Dictionary with detection results
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/detect",
                json={
                    "text": text,
                    "confidence_threshold": confidence_threshold,
                    "include_details": include_details
                },
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "data": None
            }
    
    def detect_bullying_batch(self, texts: List[str], confidence_threshold: float = 0.7, include_details: bool = True) -> Dict:
        """
        Detect bullying in multiple texts
        
        Args:
            texts: List of texts to analyze
            confidence_threshold: Minimum confidence required (0.0 to 1.0)
            include_details: Whether to include detailed detection information
            
        Returns:
            Dictionary with batch detection results
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/batch-detect",
                json={
                    "texts": texts,
                    "confidence_threshold": confidence_threshold,
                    "include_details": include_details
                },
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "data": None
            }
    
    def add_bullying_words(self, words: List[str]) -> Dict:
        """
        Add new bullying words to the database
        
        Args:
            words: List of words to add
            
        Returns:
            Dictionary with addition results
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/api/add-words",
                json={"words": words},
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "data": None
            }
    
    def get_statistics(self) -> Dict:
        """
        Get API usage statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            response = requests.get(
                f"{self.api_base_url}/api/stats",
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}",
                "data": None
            }
    
    def is_server_healthy(self) -> bool:
        """
        Check if the API server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=10)
            return response.status_code == 200
        except:
            return False

def example_usage():
    """Demonstrate how to use the client"""
    
    # Initialize client
    client = BullyingDetectionClient()
    
    # Check if server is running
    if not client.is_server_healthy():
        print("Error: API server is not running or not healthy")
        print("Please start the server first with: python app.py")
        return
    
    print("Cyberbullying Detection Client Example")
    print("=" * 40)
    
    # Example 1: Single text detection
    print("\n1. Single Text Detection:")
    test_text = "You are such a loser, nobody likes you!"
    result = client.detect_bullying(test_text)
    
    if result["success"]:
        data = result["data"]
        print(f"Text: '{test_text}'")
        print(f"Is Bullying: {data['is_bullying']}")
        print(f"Confidence: {data['confidence']:.2f}")
        print(f"Severity: {data['severity']}")
        print(f"Detection Method: {data['detection_method']}")
    else:
        print(f"Error: {result['error']}")
    
    # Example 2: Batch detection
    print("\n2. Batch Detection:")
    test_texts = [
        "Hello, how are you?",
        "You are stupid!",
        "Great work on the project!",
        "I hate you so much, kill yourself"
    ]
    
    batch_result = client.detect_bullying_batch(test_texts, include_details=False)
    
    if batch_result["success"]:
        results = batch_result["data"]["results"]
        print(f"Processed {len(results)} texts:")
        for result in results:
            idx = result["index"]
            print(f"  [{idx}] '{test_texts[idx]}' -> {result['is_bullying']} ({result['confidence']:.2f})")
    else:
        print(f"Error: {batch_result['error']}")
    
    # Example 3: Get statistics
    print("\n3. API Statistics:")
    stats_result = client.get_statistics()
    
    if stats_result["success"]:
        stats = stats_result["data"]
        print(f"Total detections: {stats['total_detections']}")
        print(f"Bullying found: {stats['bullying_found']}")
        print(f"Local vs OpenAI: {stats['local_detections']} / {stats['openai_detections']}")
    else:
        print(f"Error: {stats_result['error']}")
    
    # Example 4: Add custom words
    print("\n4. Adding Custom Words:")
    new_words = ["newbullword1", "newbullword2"]
    add_result = client.add_bullying_words(new_words)
    
    if add_result["success"]:
        print(f"Added {add_result['data']['words_added']} new words")
    else:
        print(f"Error: {add_result['error']}")

def interactive_mode():
    """Interactive mode for testing the API"""
    client = BullyingDetectionClient()
    
    if not client.is_server_healthy():
        print("Error: API server is not running")
        return
    
    print("Interactive Cyberbullying Detection")
    print("Type 'quit' to exit")
    print("-" * 30)
    
    while True:
        try:
            text = input("\nEnter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = client.detect_bullying(text, include_details=False)
            
            if result["success"]:
                data = result["data"]
                status = "🚨 BULLYING DETECTED" if data['is_bullying'] else "✅ Safe content"
                print(f"{status}")
                print(f"Confidence: {data['confidence']:.2f}")
                print(f"Severity: {data['severity']}")
            else:
                print(f"Error: {result['error']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        example_usage()
