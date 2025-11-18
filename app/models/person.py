"""
Person model - represents a celebrity/person in the training dataset.
"""

from app.database import db
from datetime import datetime


class Person(db.Model):
    """
    Represents a person (celebrity) in the training system.
    """

    __tablename__ = 'people'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Domain reference (which dataset this person belongs to)
    domain = db.Column(db.String(50), db.ForeignKey('domains.domain_code'), nullable=False, index=True)

    # Identity
    full_name = db.Column(db.String(255), nullable=False)
    normalized_name = db.Column(db.String(255), nullable=False, index=True)  # Folder name

    # Wikidata information
    wikidata_id = db.Column(db.String(50), index=True)
    primary_image_url = db.Column(db.Text)
    sitelinks = db.Column(db.Integer, default=0)  # Notability metric

    # Metadata
    occupation = db.Column(db.String(100))
    country = db.Column(db.String(50))
    description = db.Column(db.Text)

    # Training status
    status = db.Column(db.String(50), default='pending', index=True)  # pending, in_training, completed, deployed
    folder_path = db.Column(db.String(500))

    # Image statistics
    total_images = db.Column(db.Integer, default=0)
    images_from_wikimedia = db.Column(db.Integer, default=0)
    images_from_serp = db.Column(db.Integer, default=0)
    images_validated = db.Column(db.Integer, default=0)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    trained_at = db.Column(db.DateTime)
    deployed_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    domain_ref = db.relationship('Domain', back_populates='people')
    images = db.relationship('Image', back_populates='person', lazy='dynamic', cascade='all, delete-orphan')

    # Composite unique constraint: normalized_name must be unique per domain
    __table_args__ = (
        db.UniqueConstraint('domain', 'normalized_name', name='uq_domain_person'),
        db.Index('idx_domain_status', 'domain', 'status'),
    )

    def to_dict(self, include_images=False):
        """Convert to dictionary for API responses"""
        data = {
            'id': self.id,
            'domain': self.domain,
            'full_name': self.full_name,
            'normalized_name': self.normalized_name,
            'wikidata_id': self.wikidata_id,
            'sitelinks': self.sitelinks,
            'occupation': self.occupation,
            'country': self.country,
            'description': self.description,
            'status': self.status,
            'total_images': self.total_images,
            'images_from_wikimedia': self.images_from_wikimedia,
            'images_from_serp': self.images_from_serp,
            'images_validated': self.images_validated,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None
        }

        if include_images:
            data['images'] = [img.to_dict() for img in self.images.all()]

        return data

    def __repr__(self):
        return f'<Person {self.full_name} ({self.domain})>'
