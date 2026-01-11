from app import create_app
# from flask_cors import CORS  # Disabled - CORS is handled by nginx

app = create_app()

# CORS handled by nginx to avoid duplicate headers
# CORS(app, resources={
#     r"/*": {
#         "origins": [
#             "https://photolytics.mpanel.app",  # Production frontend
#             "http://localhost:5173",           # Local development (Vite)
#             "http://localhost:3000",           # Local development (React)
#         ],
#         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept"],
#         "expose_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True,
#         "max_age": 3600
#     }
# })

if __name__ == '__main__':
    app.run(debug=True)
