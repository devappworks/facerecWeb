from app import create_app
from flask_cors import CORS

app = create_app()

# CORS konfiguracija - dozvoljava pristup sa produkcione frontend aplikacije
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://photolytics.mpanel.app",  # Production frontend
            "http://localhost:5173",           # Local development (Vite)
            "http://localhost:3000",           # Local development (React)
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

if __name__ == '__main__':
    app.run(debug=True) 