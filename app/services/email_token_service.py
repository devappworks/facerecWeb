from dotenv import load_dotenv
import os
import json
from typing import Dict, Optional, Tuple, List

load_dotenv()


class EmailTokenService:
    """Service for handling email-to-token mapping operations with multi-domain support."""
    
    def __init__(self):
        """Initialize the service with email-to-key mapping and client tokens."""
        try:
            # CLIENTS_EMAILS can now map email -> list of keys for different domains
            # {"rts@gmail.com": ["rts", "rts_domain2"], "hrt@gmail.com": ["hrt"], ...}
            # or backwards compatible: {"rts@gmail.com": "rts", ...}
            self.clients_emails: Dict[str, any] = json.loads(
                os.getenv('CLIENTS_EMAILS', '{}')
            )
            
            # CLIENTS_TOKENS mapira token -> key  
            # {"dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD": "rts", ...}
            self.clients_tokens: Dict[str, str] = json.loads(
                os.getenv('CLIENTS_TOKENS', '{}')
            )
            
            # Kreiramo inverzni mapiranje: key -> token
            self.key_to_token: Dict[str, str] = {}
            for token, key in self.clients_tokens.items():
                self.key_to_token[key] = token
                
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON configuration in environment variables: {str(e)}")
    
    def get_tokens_by_email(self, email: str) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
        """
        Get all tokens for given email address (supports multiple domains).
        
        Args:
            email (str): Email address to look up
            
        Returns:
            Tuple[Optional[List[Dict[str, str]]], Optional[str]]: (list_of_token_data, error_message)
            If successful, returns ([{"token": "...", "email": "...", "domain": "..."}], None)
            If error, returns (None, error_message)
        """
        if not email or not isinstance(email, str):
            return None, "Email address is required and must be a valid string"
        
        email = email.strip().lower()
        
        # Check if email exists in CLIENTS_EMAILS
        if email not in self.clients_emails:
            return None, f"Email '{email}' not found in authorized users"
        
        # Get the key(s) for this email - support both single key and list of keys
        keys_data = self.clients_emails[email]
        
        # Handle backwards compatibility - if it's a string, convert to list
        if isinstance(keys_data, str):
            keys = [keys_data]
        elif isinstance(keys_data, list):
            keys = keys_data
        else:
            return None, f"Invalid key configuration for email '{email}'"
        
        # Collect all tokens for this email
        tokens_data = []
        missing_keys = []
        
        for key in keys:
            if key in self.key_to_token:
                tokens_data.append({
                    "token": self.key_to_token[key],
                    "email": email,
                    "domain": key  # Using key as domain identifier
                })
            else:
                missing_keys.append(key)
        
        # If we have missing keys, return error
        if missing_keys:
            return None, f"Token not found for key(s) '{', '.join(missing_keys)}'. Please contact administrator"
        
        # If no tokens found at all
        if not tokens_data:
            return None, f"No tokens found for email '{email}'. Please contact administrator"
        
        return tokens_data, None
    
    def get_token_by_email(self, email: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get token for given email address (backwards compatible - returns first token).
        
        Args:
            email (str): Email address to look up
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (token, error_message)
            If successful, returns (token, None)
            If error, returns (None, error_message)
        """
        tokens_data, error_message = self.get_tokens_by_email(email)
        
        if error_message:
            return None, error_message
        
        if tokens_data and len(tokens_data) > 0:
            return tokens_data[0]["token"], None
        
        return None, f"No token found for email '{email}'"

    def validate_email_exists(self, email: str) -> bool:
        """
        Check if email exists in the mapping.
        
        Args:
            email (str): Email address to validate
            
        Returns:
            bool: True if email exists, False otherwise
        """
        if not email or not isinstance(email, str):
            return False
        
        return email.strip().lower() in self.clients_emails
    
    def get_all_authorized_emails(self) -> list:
        """
        Get list of all authorized email addresses.
        
        Returns:
            list: List of authorized email addresses
        """
        return list(self.clients_emails.keys())
    
    def get_key_by_email(self, email: str) -> Optional[str]:
        """
        Get key for given email address (backwards compatible - returns first key).
        
        Args:
            email (str): Email address to look up
            
        Returns:
            Optional[str]: Key if found, None otherwise
        """
        if not email or not isinstance(email, str):
            return None
        
        email = email.strip().lower()
        keys_data = self.clients_emails.get(email)
        
        if isinstance(keys_data, str):
            return keys_data
        elif isinstance(keys_data, list) and len(keys_data) > 0:
            return keys_data[0]
        
        return None