#!/usr/bin/env python3
"""
Test batch training with 5 persons using improved SERP consensus.
Tests the new configuration:
- No P18 fallback
- 10 SERP images for consensus (was 4)
- 5/10 consensus requirement (was 3/4)
- 0.50 distance threshold (was 0.45)
"""

import os
import sys
import json
import logging
from datetime import datetime

# Setup detailed logging
log_filename = f"storage/logs/batch_5_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("storage/logs", exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Reduce noise from some libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

logger.info("="*70)
logger.info("BATCH 5 IMPROVED TEST - Starting")
logger.info("Configuration:")
logger.info("  - P18 fallback: DISABLED")
logger.info("  - SERP consensus images: 10 (was 4)")
logger.info("  - Consensus requirement: 5/10 (was 3/4)")
logger.info("  - Distance threshold: 0.50 (was 0.45)")
logger.info(f"Log file: {log_filename}")
logger.info("="*70)

# Import after logging setup
from app import create_app
from app.services.automated_training_service import AutomatedTrainingService
from app.services.wikidata_service import WikidataService

app = create_app()

with app.app_context():
    # Initialize services
    logger.info("Initializing services...")
    training_service = AutomatedTrainingService(domain='serbia')
    wikidata_service = WikidataService()

    # Get 5 candidates from Wikidata
    logger.info("Fetching candidates from Wikidata...")
    # Query basketball players from Serbia
    all_candidates = WikidataService.query_celebrities(country='serbia', occupation='basketball_player', limit=50)
    # Use different range to avoid previous test subjects
    candidates = all_candidates[30:35] if len(all_candidates) > 35 else all_candidates[:5]

    logger.info(f"Got {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        logger.info(f"  {i+1}. {c.get('full_name')}")

    # Process each person
    results = []
    for i, candidate in enumerate(candidates):
        full_name = candidate.get('full_name', '')
        person_name = full_name
        wikidata_id = candidate.get('wikidata_id', '')
        # Note: P18 URL present but will NOT be used (fallback disabled)
        p18_url = candidate.get('image_url') or candidate.get('p18_url')
        search_name = full_name

        logger.info("")
        logger.info("="*70)
        logger.info(f"PERSON {i+1}/5: {person_name}")
        logger.info(f"  Wikidata ID: {wikidata_id}")
        logger.info(f"  P18 available: {'Yes' if p18_url else 'No'} (will NOT be used)")
        logger.info("="*70)

        try:
            # Generate unique batch ID for this person
            batch_id = f"improved5_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{training_service._safe_folder_name(person_name)}"

            # Process the person using internal method
            result = training_service._process_person(
                batch_id=batch_id,
                person_name=person_name,
                wikidata_id=wikidata_id,
                p18_url=p18_url,
                images_per_person=10,  # Target 10 images per person
                search_name=search_name
            )

            results.append({
                'name': person_name,
                'batch_id': batch_id,
                'images_found': result.get('images_found', 0),
                'images_accepted': result.get('images_accepted', 0),
                'success': result.get('images_accepted', 0) >= 5,  # At least 5 images = success
                'gallery_url': result.get('gallery_url')
            })

            logger.info(f"RESULT for {person_name}:")
            logger.info(f"  Images found: {result.get('images_found', 0)}")
            logger.info(f"  Images accepted: {result.get('images_accepted', 0)}")
            logger.info(f"  Success: {result.get('images_accepted', 0) >= 5}")

        except Exception as e:
            logger.error(f"ERROR processing {person_name}: {str(e)}", exc_info=True)
            results.append({
                'name': person_name,
                'error': str(e),
                'success': False
            })

    # Print summary
    logger.info("")
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    successful = sum(1 for r in results if r.get('success'))
    total_found = sum(r.get('images_found', 0) for r in results)
    total_accepted = sum(r.get('images_accepted', 0) for r in results)

    logger.info(f"Total persons: {len(results)}")
    logger.info(f"Successful (>=5 images): {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    logger.info(f"Total images found: {total_found}")
    logger.info(f"Total images accepted: {total_accepted}")
    if total_found > 0:
        logger.info(f"Acceptance rate: {total_accepted/total_found*100:.1f}%")

    logger.info("")
    logger.info("Per-person results:")
    for r in results:
        status = "✓" if r.get('success') else "✗"
        if 'error' in r:
            logger.info(f"  {status} {r['name']}: ERROR - {r['error']}")
        else:
            logger.info(f"  {status} {r['name']}: {r.get('images_accepted', 0)}/{r.get('images_found', 0)} accepted")

    logger.info("")
    logger.info("="*70)
    logger.info("COMPARISON WITH PREVIOUS TEST (10 people, old config):")
    logger.info("  OLD: 30% success rate (3/10), 22.9% acceptance rate")
    logger.info(f"  NEW: {successful/len(results)*100:.1f}% success rate ({successful}/{len(results)}), {total_accepted/total_found*100:.1f if total_found > 0 else 0:.1f}% acceptance rate")
    logger.info("="*70)
    logger.info("")
    logger.info(f"Full log saved to: {log_filename}")
    logger.info("="*70)
