"""
Batch Test models - represents batch testing sessions and their results.
"""

from app.database import db
from datetime import datetime
import json


class BatchTestSession(db.Model):
    """
    Represents a batch testing session containing multiple image results.
    """

    __tablename__ = 'batch_test_sessions'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Session identifier (UUID)
    session_id = db.Column(db.String(36), unique=True, nullable=False, index=True)

    # User info
    user_email = db.Column(db.String(255), nullable=False, index=True)
    domain = db.Column(db.String(50), db.ForeignKey('domains.domain_code'), nullable=False, index=True)

    # Session metadata
    name = db.Column(db.String(255))  # Optional user-defined name
    model_used = db.Column(db.String(100))  # AI model used for analysis
    total_images = db.Column(db.Integer, default=0)
    successful_count = db.Column(db.Integer, default=0)
    failed_count = db.Column(db.Integer, default=0)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime)

    # Relationships
    results = db.relationship('BatchTestResult', back_populates='session', cascade='all, delete-orphan')

    def to_dict(self, include_results=False):
        """Convert to dictionary for API responses"""
        data = {
            'id': self.id,
            'session_id': self.session_id,
            'user_email': self.user_email,
            'domain': self.domain,
            'name': self.name,
            'model_used': self.model_used,
            'total_images': self.total_images,
            'successful_count': self.successful_count,
            'failed_count': self.failed_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
        if include_results:
            data['results'] = [r.to_dict() for r in self.results]
        return data

    def __repr__(self):
        return f'<BatchTestSession {self.session_id} by {self.user_email}>'


class BatchTestResult(db.Model):
    """
    Represents a single image result within a batch testing session.
    """

    __tablename__ = 'batch_test_results'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Foreign key to session
    session_id = db.Column(db.Integer, db.ForeignKey('batch_test_sessions.id', ondelete='CASCADE'), nullable=False, index=True)

    # Image info
    filename = db.Column(db.String(255), nullable=False)
    image_thumbnail = db.Column(db.Text)  # Base64 thumbnail (small, for display)

    # Analysis results
    model_used = db.Column(db.String(100))
    status = db.Column(db.String(20), nullable=False)  # 'success', 'error'
    error_message = db.Column(db.Text)

    # Face recognition results (from face DB)
    recognized_persons = db.Column(db.Text)  # JSON array of {name, confidence}

    # Gemini AI recognition results
    identified_persons = db.Column(db.Text)  # JSON array of {name, confidence}

    # Description
    description = db.Column(db.Text)

    # Full metadata (JSON)
    full_metadata = db.Column(db.Text)  # Complete API response metadata

    # Timestamp
    processed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = db.relationship('BatchTestSession', back_populates='results')

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'filename': self.filename,
            'image_thumbnail': self.image_thumbnail,
            'model_used': self.model_used,
            'status': self.status,
            'error_message': self.error_message,
            'recognized_persons': json.loads(self.recognized_persons) if self.recognized_persons else [],
            'identified_persons': json.loads(self.identified_persons) if self.identified_persons else [],
            'description': self.description,
            'full_metadata': json.loads(self.full_metadata) if self.full_metadata else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }

    def __repr__(self):
        return f'<BatchTestResult {self.filename} status={self.status}>'
