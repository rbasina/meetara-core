#!/usr/bin/env python3
"""
Simple script to test the Meetara Chat API
"""

import requests
import json

def test_chat_api():
    """Test the chat API with various queries."""
    
    base_url = "http://localhost:8000"
    
    print("Testing Meetara Chat API")
    print("=" * 50)
    
    # Test 1: Basic greeting
    print("\n1. Testing basic greeting...")
    response = requests.post(
        f"{base_url}/api/chat/",
        json={
            "query": "Hello, how are you?",
            "context": {}
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS - Response: {data['response']}")
        print(f"Domain: {data['domain']}")
        print(f"Confidence: {data['confidence']}")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")
    
    # Test 2: Health-related query
    print("\n2. Testing health query...")
    response = requests.post(
        f"{base_url}/api/chat/",
        json={
            "query": "I'm feeling stressed about my health",
            "context": {"emotion": "anxious"}
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS - Response: {data['response']}")
        print(f"Domain: {data['domain']}")
        print(f"Emotion Context: {data.get('emotion_context', 'None')}")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")
    
    # Test 3: Programming query
    print("\n3. Testing programming query...")
    response = requests.post(
        f"{base_url}/api/chat/",
        json={
            "query": "How do I write a Python function?",
            "context": {}
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS - Response: {data['response']}")
        print(f"Domain: {data['domain']}")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")
    
    # Test 4: Check available domains
    print("\n4. Checking available domains...")
    response = requests.get(f"{base_url}/api/chat/domains")
    
    if response.status_code == 200:
        domains = response.json()
        print(f"SUCCESS - Available domains: {len(domains)}")
        for domain in domains:
            print(f"   - {domain}")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")
    
    # Test 5: Health check
    print("\n5. Checking system health...")
    response = requests.get(f"{base_url}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"SUCCESS - System Status: {health['status']}")
        print(f"Components: {health['components']}")
    else:
        print(f"ERROR: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_chat_api()
