"""
Database configuration and initialization.
Uses SQLAlchemy with SQLite for simple, file-based storage.
"""

from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy instance
db = SQLAlchemy()


def init_db(app):
    """
    Initialize database with Flask app.

    Args:
        app: Flask application instance
    """
    db.init_app(app)

    with app.app_context():
        # Import models to ensure they're registered
        from app.models import Domain, Person, Image, BatchTestSession, BatchTestResult

        # Create all tables
        db.create_all()

        # Initialize default domain if not exists
        _init_default_domain()


def _init_default_domain():
    """Initialize default 'serbia' domain for backward compatibility"""
    from app.models.domain import Domain

    if not Domain.query.filter_by(domain_code='serbia').first():
        default_domain = Domain(
            domain_code='serbia',
            display_name='Serbia',
            default_country='serbia',
            is_active=True
        )
        db.session.add(default_domain)
        db.session.commit()
