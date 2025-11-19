"""
TrainingSession model - represents batch training operations.
"""

from app.database import db
from datetime import datetime


class TrainingSession(db.Model):
    """
    Represents a training batch operation.
    Tracks overall progress, statistics, and costs.
    """

    __tablename__ = 'training_sessions'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Session identity
    session_id = db.Column(db.String(50), unique=True, nullable=False, index=True)  # UUID

    # Domain reference
    domain = db.Column(db.String(50), db.ForeignKey('domains.domain_code'), nullable=False, index=True)

    # Session parameters
    country = db.Column(db.String(50))
    occupation = db.Column(db.String(100))

    # Progress tracking
    status = db.Column(db.String(50), default='processing', index=True)  # processing, completed, failed
    total_people = db.Column(db.Integer, default=0)
    people_completed = db.Column(db.Integer, default=0)
    people_failed = db.Column(db.Integer, default=0)

    # Image statistics
    total_images_downloaded = db.Column(db.Integer, default=0)
    images_from_wikimedia = db.Column(db.Integer, default=0)
    images_from_serp = db.Column(db.Integer, default=0)

    # Cost tracking (for ROI analysis)
    estimated_serp_cost = db.Column(db.Numeric(10, 2), default=0.0)

    # Timestamps
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime)

    # Error tracking
    error_message = db.Column(db.Text)

    # Relationships
    domain_ref = db.relationship('Domain', back_populates='training_sessions')

    # Indexes
    __table_args__ = (
        db.Index('idx_domain_status', 'domain', 'status'),
    )

    def to_dict(self):
        """Convert to dictionary for API responses"""
        # Calculate statistics
        wikimedia_percentage = 0
        if self.total_images_downloaded > 0:
            wikimedia_percentage = (self.images_from_wikimedia / self.total_images_downloaded) * 100

        progress_percentage = 0
        if self.total_people > 0:
            progress_percentage = (self.people_completed / self.total_people) * 100

        return {
            'id': self.id,
            'session_id': self.session_id,
            'domain': self.domain,
            'country': self.country,
            'occupation': self.occupation,
            'status': self.status,
            'progress': {
                'total_people': self.total_people,
                'completed': self.people_completed,
                'failed': self.people_failed,
                'percentage': round(progress_percentage, 1)
            },
            'images': {
                'total': self.total_images_downloaded,
                'from_wikimedia': self.images_from_wikimedia,
                'from_serp': self.images_from_serp,
                'wikimedia_percentage': round(wikimedia_percentage, 1)
            },
            'cost': {
                'estimated_serp_cost': float(self.estimated_serp_cost) if self.estimated_serp_cost else 0,
                'savings_percentage': round(wikimedia_percentage, 1)
            },
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message
        }

    def __repr__(self):
        return f'<TrainingSession {self.session_id} ({self.domain})>'
