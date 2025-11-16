import requests

BASE_URL = "http://localhost:5000"

test_words = [
    ('madarchod', True),
    ('stupid', True),
    ('I hate you', True),
    ('die', False),  # Should not trigger alone
    ('hello', False),
    ('kill yourself', True),
    ('you are worthless', True)
]

print("Quick API Test Results:")
print("=" * 40)

for word, expected in test_words:
    try:
        response = requests.post(
            f"{BASE_URL}/api/detect",
            json={"text": word}
        )
        if response.status_code == 200:
            data = response.json()['data']
            is_bullying = data['is_bullying']
            confidence = data['confidence']
            status = "✓" if is_bullying == expected else "✗"
            print(f"{status} '{word}': {is_bullying} (Conf: {confidence:.1%})")
        else:
            print(f"✗ '{word}': API Error {response.status_code}")
    except Exception as e:
        print(f"✗ '{word}': Connection Error - Is server running?")
