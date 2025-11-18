"""
Domain management API endpoints.
Handles domain CRUD operations and statistics.
"""

from flask import Blueprint, jsonify, request
from app.database import db
from app.models.domain import Domain
from app.models.person import Person
from app.models.image import Image
from app.models.training_session import TrainingSession
from sqlalchemy import func
import logging
import os

logger = logging.getLogger(__name__)

domain_bp = Blueprint('domains', __name__, url_prefix='/api/domains')


@domain_bp.route('', methods=['GET'])
def list_domains():
    """
    List all domains.

    Returns:
        JSON list of all domains with basic info
    """
    try:
        domains = Domain.query.filter_by(is_active=True).all()

        return jsonify({
            'success': True,
            'domains': [domain.to_dict() for domain in domains],
            'total': len(domains)
        }), 200

    except Exception as e:
        logger.error(f"Error listing domains: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to list domains',
            'message': str(e)
        }), 500


@domain_bp.route('/<domain_code>', methods=['GET'])
def get_domain(domain_code):
    """
    Get detailed information about a specific domain.

    Args:
        domain_code: Domain code (e.g., 'serbia')

    Returns:
        JSON with domain details and statistics
    """
    try:
        domain = Domain.query.filter_by(domain_code=domain_code).first()

        if not domain:
            return jsonify({
                'success': False,
                'error': 'Domain not found'
            }), 404

        # Get statistics
        people_count = Person.query.filter_by(domain=domain_code).count()
        images_count = Image.query.filter_by(domain=domain_code).count()

        # Get training sessions
        sessions = TrainingSession.query.filter_by(domain=domain_code).order_by(
            TrainingSession.started_at.desc()
        ).limit(5).all()

        # Calculate source breakdown
        wikimedia_count = db.session.query(func.sum(Image.file_size)).filter(
            Image.domain == domain_code,
            Image.source.like('wikimedia%')
        ).scalar() or 0

        serp_count = db.session.query(func.count(Image.id)).filter(
            Image.domain == domain_code,
            Image.source == 'serp'
        ).scalar() or 0

        result = domain.to_dict()
        result['statistics'] = {
            'people_count': people_count,
            'images_count': images_count,
            'images_from_wikimedia': wikimedia_count,
            'images_from_serp': serp_count,
            'recent_sessions': [session.to_dict() for session in sessions]
        }

        return jsonify({
            'success': True,
            'domain': result
        }), 200

    except Exception as e:
        logger.error(f"Error getting domain {domain_code}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get domain',
            'message': str(e)
        }), 500


@domain_bp.route('', methods=['POST'])
def create_domain():
    """
    Create a new domain.

    Body:
        {
            "domain_code": "greece",
            "display_name": "Greece",
            "default_country": "greece"
        }

    Returns:
        JSON with created domain info
    """
    try:
        data = request.get_json()

        # Validate required fields
        if not data or 'domain_code' not in data or 'display_name' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: domain_code, display_name'
            }), 400

        # Check if domain already exists
        existing = Domain.query.filter_by(domain_code=data['domain_code']).first()
        if existing:
            return jsonify({
                'success': False,
                'error': f"Domain '{data['domain_code']}' already exists"
            }), 409

        # Create new domain
        domain = Domain(
            domain_code=data['domain_code'],
            display_name=data['display_name'],
            default_country=data.get('default_country'),
            default_occupations=data.get('default_occupations'),
            is_active=data.get('is_active', True)
        )

        db.session.add(domain)
        db.session.commit()

        # Create directories
        os.makedirs(domain.training_path, exist_ok=True)
        os.makedirs(domain.staging_path, exist_ok=True)
        os.makedirs(domain.production_path, exist_ok=True)
        os.makedirs(domain.batched_path, exist_ok=True)

        logger.info(f"Created new domain: {domain.domain_code}")

        return jsonify({
            'success': True,
            'message': f"Domain '{domain.domain_code}' created successfully",
            'domain': domain.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating domain: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to create domain',
            'message': str(e)
        }), 500


@domain_bp.route('/<domain_code>', methods=['PUT'])
def update_domain(domain_code):
    """
    Update domain settings.

    Args:
        domain_code: Domain code

    Body:
        {
            "display_name": "Updated Name",
            "is_active": false
        }

    Returns:
        JSON with updated domain info
    """
    try:
        domain = Domain.query.filter_by(domain_code=domain_code).first()

        if not domain:
            return jsonify({
                'success': False,
                'error': 'Domain not found'
            }), 404

        data = request.get_json()

        # Update fields
        if 'display_name' in data:
            domain.display_name = data['display_name']
        if 'default_country' in data:
            domain.default_country = data['default_country']
        if 'default_occupations' in data:
            domain.default_occupations = data['default_occupations']
        if 'is_active' in data:
            domain.is_active = data['is_active']

        db.session.commit()

        logger.info(f"Updated domain: {domain.domain_code}")

        return jsonify({
            'success': True,
            'message': f"Domain '{domain.domain_code}' updated successfully",
            'domain': domain.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating domain {domain_code}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to update domain',
            'message': str(e)
        }), 500


@domain_bp.route('/<domain_code>/stats', methods=['GET'])
def get_domain_stats(domain_code):
    """
    Get detailed statistics for a domain.

    Args:
        domain_code: Domain code

    Returns:
        JSON with comprehensive statistics
    """
    try:
        domain = Domain.query.filter_by(domain_code=domain_code).first()

        if not domain:
            return jsonify({
                'success': False,
                'error': 'Domain not found'
            }), 404

        # People statistics
        people_stats = db.session.query(
            Person.status,
            func.count(Person.id).label('count')
        ).filter(
            Person.domain == domain_code
        ).group_by(Person.status).all()

        # Image statistics by source
        image_stats = db.session.query(
            Image.source,
            func.count(Image.id).label('count')
        ).filter(
            Image.domain == domain_code
        ).group_by(Image.source).all()

        # Training sessions summary
        session_stats = db.session.query(
            TrainingSession.status,
            func.count(TrainingSession.id).label('count'),
            func.sum(TrainingSession.total_images_downloaded).label('total_images'),
            func.sum(TrainingSession.images_from_wikimedia).label('wikimedia_images'),
            func.sum(TrainingSession.images_from_serp).label('serp_images')
        ).filter(
            TrainingSession.domain == domain_code
        ).group_by(TrainingSession.status).all()

        # Calculate cost savings
        total_images = sum(stat.total_images or 0 for stat in session_stats)
        wikimedia_images = sum(stat.wikimedia_images or 0 for stat in session_stats)
        cost_savings_percentage = (wikimedia_images / total_images * 100) if total_images > 0 else 0

        return jsonify({
            'success': True,
            'domain': domain_code,
            'statistics': {
                'people': {
                    status: count for status, count in people_stats
                },
                'images': {
                    'by_source': {
                        source: count for source, count in image_stats
                    },
                    'total': sum(count for _, count in image_stats)
                },
                'training_sessions': {
                    status: {
                        'count': count,
                        'total_images': total_images or 0,
                        'wikimedia_images': wikimedia or 0,
                        'serp_images': serp or 0
                    } for status, count, total_images, wikimedia, serp in session_stats
                },
                'cost_savings': {
                    'total_images': total_images,
                    'wikimedia_images': wikimedia_images,
                    'serp_images': total_images - wikimedia_images if total_images > 0 else 0,
                    'savings_percentage': round(cost_savings_percentage, 2)
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting stats for domain {domain_code}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get domain statistics',
            'message': str(e)
        }), 500
