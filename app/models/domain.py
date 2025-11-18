"""
Domain model - represents separate training datasets per client/region.
"""

from app.database import db
from datetime import datetime


class Domain(db.Model):
    """
    Represents a domain (client/region) with separate training dataset.
    Examples: 'serbia', 'greece', 'slovenia', 'sports_global'
    """

    __tablename__ = 'domains'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Domain identity
    domain_code = db.Column(db.String(50), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(100), nullable=False)

    # Default training parameters
    default_country = db.Column(db.String(50))
    default_occupations = db.Column(db.Text)  # JSON array: ["actor", "athlete"]

    # Paths (auto-generated from domain_code)
    training_path = db.Column(db.String(500))
    staging_path = db.Column(db.String(500))
    production_path = db.Column(db.String(500))
    batched_path = db.Column(db.String(500))

    # Statistics
    total_people = db.Column(db.Integer, default=0)
    total_images = db.Column(db.Integer, default=0)

    # Status
    is_active = db.Column(db.Boolean, default=True, index=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    people = db.relationship('Person', back_populates='domain_ref', lazy='dynamic', cascade='all, delete-orphan')
    training_sessions = db.relationship('TrainingSession', back_populates='domain_ref', lazy='dynamic')

    def __init__(self, **kwargs):
        super(Domain, self).__init__(**kwargs)
        # Auto-generate paths from domain_code
        if self.domain_code and not self.training_path:
            self._generate_paths()

    def _generate_paths(self):
        """Generate storage paths from domain_code"""
        self.training_path = f'storage/training/{self.domain_code}'
        self.staging_path = f'storage/trainingPass/{self.domain_code}'
        self.production_path = f'storage/recognized_faces_prod/{self.domain_code}'
        self.batched_path = f'storage/recognized_faces_batched/{self.domain_code}'

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'domain_code': self.domain_code,
            'display_name': self.display_name,
            'default_country': self.default_country,
            'total_people': self.total_people,
            'total_images': self.total_images,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'paths': {
                'training': self.training_path,
                'staging': self.staging_path,
                'production': self.production_path,
                'batched': self.batched_path
            }
        }

    def __repr__(self):
        return f'<Domain {self.domain_code}>'
