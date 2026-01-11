from flask import Flask
# from flask_cors import CORS
from config import Config
from app.routes.image_routes import image_routes
from app.routes.admin_routes import admin_routes
from app.routes.excel_routes import excel_bp
from app.routes.auth_routes import auth_routes
from app.routes.batch_recognition_routes import batch_recognition_bp
from app.routes.test_recognition_routes import test_recognition_routes
from app.routes.video_routes import video_bp
from app.routes.domain_routes import domain_bp
from app.routes.storage_routes import storage_bp
from app.routes.training_routes import training_bp
from app.routes.batch_test_routes import batch_test_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Povećanje maksimalne veličine zahteva na 100MB (for video uploads)
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

    #     # Initialize CORS
    #     CORS(app)

    # Initialize database
    from app.database import init_db
    init_db(app)

    # Registrujemo rute
    app.register_blueprint(image_routes)
    app.register_blueprint(admin_routes, url_prefix='/admin')
    app.register_blueprint(excel_bp)
    app.register_blueprint(auth_routes)
    app.register_blueprint(batch_recognition_bp)
    app.register_blueprint(test_recognition_routes)
    app.register_blueprint(video_bp)
    app.register_blueprint(domain_bp)  # Domain management endpoints
    app.register_blueprint(storage_bp)  # Storage management endpoints
    app.register_blueprint(training_bp)  # Automated training endpoints
    app.register_blueprint(batch_test_bp)  # Batch testing endpoints

    return app