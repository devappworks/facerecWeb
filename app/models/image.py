"""
Image model - represents individual training images.
"""

from app.database import db
from datetime import datetime


class Image(db.Model):
    """
    Represents an individual training image for a person.
    """

    __tablename__ = 'images'

    # Primary key
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    person_id = db.Column(db.Integer, db.ForeignKey('people.id', ondelete='CASCADE'), nullable=False, index=True)
    domain = db.Column(db.String(50), db.ForeignKey('domains.domain_code'), nullable=False, index=True)

    # File information
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)  # Bytes

    # Source tracking
    source = db.Column(db.String(50), nullable=False, index=True)  # 'wikimedia_p18', 'wikimedia_category', 'serp'
    source_url = db.Column(db.Text)  # Original download URL

    # Quality metrics
    is_validated = db.Column(db.Boolean, default=False, index=True)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    validation_error = db.Column(db.Text)  # Why validation failed (if applicable)

    # Timestamps
    downloaded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    validated_at = db.Column(db.DateTime)

    # Relationships
    person = db.relationship('Person', back_populates='images')

    # Indexes
    __table_args__ = (
        db.Index('idx_person_source', 'person_id', 'source'),
        db.Index('idx_domain_validated', 'domain', 'is_validated'),
    )

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'person_id': self.person_id,
            'domain': self.domain,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'source': self.source,
            'source_url': self.source_url,
            'is_validated': self.is_validated,
            'dimensions': f"{self.width}x{self.height}" if self.width and self.height else None,
            'downloaded_at': self.downloaded_at.isoformat() if self.downloaded_at else None,
            'validated_at': self.validated_at.isoformat() if self.validated_at else None
        }

    def __repr__(self):
        return f'<Image {self.filename} for person_id={self.person_id}>'
