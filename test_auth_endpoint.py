#!/usr/bin/env python3
"""
Test script for the new authentication endpoint.

This script tests the authentication endpoint with your existing environment variables:
- CLIENTS_EMAILS: maps email -> key
- CLIENTS_TOKENS: maps token -> key

Your current environment variables:
CLIENTS_EMAILS={"rts@gmail.com": "rts", "hrt@gmail.com": "hrt", "test@gmail.com": "test", "nikola1jankovic@gmail.com": "media24"}
CLIENTS_TOKENS={"dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD": "rts", "K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh": "hrt", "EiasPl9oJWe7Ps6j5AW94DA5IXqaGCh2Seg": "test", "d9OLEFYdx18bUTGkIpaKyDFCcko1jYu0Ha1": "media24"}
"""

import requests
import json
import sys
import os

# Base URL for your Flask application
BASE_URL = "http://localhost:5000"

def test_token_by_email(email: str):
    """Test the token-by-email endpoint."""
    url = f"{BASE_URL}/api/auth/token-by-email"
    headers = {'Content-Type': 'application/json'}
    data = {'email': email}
    
    print(f"\n--- Testing token retrieval for: {email} ---")
    print(f"POST {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code, response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")
        return response.status_code, response.text

def test_validate_email(email: str):
    """Test the email validation endpoint."""
    url = f"{BASE_URL}/api/auth/validate-email"
    headers = {'Content-Type': 'application/json'}
    data = {'email': email}
    
    print(f"\n--- Testing email validation for: {email} ---")
    print(f"POST {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code, response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")
        return response.status_code, response.text

def test_health_check():
    """Test the health check endpoint."""
    url = f"{BASE_URL}/api/auth/health"
    
    print(f"\n--- Testing health check ---")
    print(f"GET {url}")
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code, response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response.text}")
        return response.status_code, response.text

def display_expected_mappings():
    """Display the expected token mappings based on your environment variables."""
    print("\n" + "="*60)
    print("EXPECTED TOKEN MAPPINGS")
    print("="*60)
    
    expected_mappings = {
        "rts@gmail.com": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
        "hrt@gmail.com": "K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh", 
        "test@gmail.com": "EiasPl9oJWe7Ps6j5AW94DA5IXqaGCh2Seg",
        "nikola1jankovic@gmail.com": "d9OLEFYdx18bUTGkIpaKyDFCcko1jYu0Ha1"
    }
    
    for email, token in expected_mappings.items():
        print(f"{email:<30} -> {token}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTHENTICATION ENDPOINT TESTS")
    print("=" * 60)
    
    # Check environment variables
    clients_emails = os.getenv('CLIENTS_EMAILS')
    clients_tokens = os.getenv('CLIENTS_TOKENS')
    
    print(f"\nEnvironment Variables:")
    print(f"CLIENTS_EMAILS: {clients_emails}")
    print(f"CLIENTS_TOKENS: {clients_tokens}")
    
    if not clients_emails or not clients_tokens:
        print("\n⚠️  WARNING: Environment variables not set!")
        print("Please make sure CLIENTS_EMAILS and CLIENTS_TOKENS are set in your environment.")
    
    display_expected_mappings()
    
    # Test health check first
    test_health_check()
    
    # Test cases based on your environment variables
    valid_emails = [
        "rts@gmail.com",
        "hrt@gmail.com", 
        "test@gmail.com",
        "nikola1jankovic@gmail.com"
    ]
    
    invalid_emails = [
        "nonexistent@gmail.com",
        "invalid.email",
        "",
        "admin@test.com"
    ]
    
    edge_case_emails = [
        "RTS@GMAIL.COM",  # Test case sensitivity
        "  rts@gmail.com  ",  # Test whitespace handling
        "NIKOLA1JANKOVIC@GMAIL.COM"  # Test mixed case
    ]
    
    # Test token retrieval for valid emails
    print(f"\n{'='*20} VALID EMAIL TOKEN TESTS {'='*20}")
    for email in valid_emails:
        test_token_by_email(email)
    
    # Test token retrieval for invalid emails
    print(f"\n{'='*20} INVALID EMAIL TOKEN TESTS {'='*20}")
    for email in invalid_emails:
        test_token_by_email(email)
    
    # Test edge cases
    print(f"\n{'='*20} EDGE CASE TOKEN TESTS {'='*20}")
    for email in edge_case_emails:
        test_token_by_email(email)
    
    # Test email validation for all cases
    print(f"\n{'='*20} EMAIL VALIDATION TESTS {'='*20}")
    all_test_emails = valid_emails + invalid_emails + edge_case_emails
    for email in all_test_emails:
        test_validate_email(email)
    
    print(f"\n{'='*60}")
    print("TESTS COMPLETED")
    print("="*60)
    print("\nExpected Results:")
    print("✅ Valid emails should return their corresponding tokens")
    print("❌ Invalid emails should return 404 error")
    print("✅ Edge case emails (different case/whitespace) should work if email exists")

if __name__ == "__main__":
    main() 