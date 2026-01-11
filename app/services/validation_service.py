from dotenv import load_dotenv
import os
import json

load_dotenv()

class ValidationService:
    def __init__(self):
        self.clients = json.loads(os.getenv('CLIENTS_TOKENS'))
        self.domain = None

    def validate_auth_token(self, auth_token):
        if auth_token in self.clients:
            self.domain = self.clients[auth_token]
            return True
        return False

    def get_domain(self):
        return self.domain
