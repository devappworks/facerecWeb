#!/usr/bin/env python
"""
Migration script: File-based system → Multi-domain architecture

This script:
1. Migrates storage/trainingPassSerbia → storage/trainingPass/serbia
2. Creates 'serbia' domain in database
3. Scans existing folders and populates database
4. Preserves all existing files

Usage:
    python migrations/migrate_to_multi_domain.py [--dry-run]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.database import db
from app.models import Domain, Person


def migrate_folders(dry_run=False):
    """Migrate folder structure to domain-based layout"""
    print("\n=== Phase 1: Migrating Folder Structure ===\n")

    migrations = [
        {
            'source': 'storage/trainingPassSerbia',
            'target': 'storage/trainingPass/serbia',
            'description': 'Training staging area'
        },
        {
            'source': 'storage/training/serbia',
            'target': 'storage/training/serbia',
            'description': 'Raw training data (already correct)'
        }
    ]

    for migration in migrations:
        source = migration['source']
        target = migration['target']
        desc = migration['description']

        print(f"📁 {desc}")
        print(f"   Source: {source}")
        print(f"   Target: {target}")

        if not os.path.exists(source):
            print(f"   ⚠️  Source doesn't exist, skipping\n")
            continue

        if source == target:
            print(f"   ✅ Already in correct location\n")
            continue

        if os.path.exists(target):
            print(f"   ⚠️  Target already exists, skipping to avoid overwrite\n")
            continue

        if dry_run:
            print(f"   🔍 DRY RUN: Would move {source} → {target}\n")
        else:
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.move(source, target)
                print(f"   ✅ Migrated successfully\n")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")

    # Handle name_mapping.json
    print("📝 Name mapping file")
    old_mapping = 'storage/name_mapping.json'
    new_mapping = 'storage/trainingPass/serbia/name_mapping.json'

    if os.path.exists(old_mapping):
        if dry_run:
            print(f"   🔍 DRY RUN: Would move {old_mapping} → {new_mapping}\n")
        else:
            try:
                os.makedirs(os.path.dirname(new_mapping), exist_ok=True)
                shutil.move(old_mapping, new_mapping)
                print(f"   ✅ Moved to {new_mapping}\n")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")
    else:
        print(f"   ℹ️  Not found, skipping\n")


def create_default_domain(app, dry_run=False):
    """Create default 'serbia' domain"""
    print("\n=== Phase 2: Creating Default Domain ===\n")

    with app.app_context():
        existing = Domain.query.filter_by(domain_code='serbia').first()

        if existing:
            print("✅ Domain 'serbia' already exists\n")
            return

        if dry_run:
            print("🔍 DRY RUN: Would create 'serbia' domain in database\n")
            return

        domain = Domain(
            domain_code='serbia',
            display_name='Serbia',
            default_country='serbia',
            is_active=True
        )

        db.session.add(domain)
        db.session.commit()

        print(f"✅ Created domain 'serbia'")
        print(f"   Training path: {domain.training_path}")
        print(f"   Staging path: {domain.staging_path}")
        print(f"   Production path: {domain.production_path}")
        print(f"   Batched path: {domain.batched_path}\n")


def scan_and_populate_database(app, dry_run=False):
    """Scan existing folders and populate database"""
    print("\n=== Phase 3: Populating Database ===\n")

    staging_path = 'storage/trainingPass/serbia'

    if not os.path.exists(staging_path):
        print(f"⚠️  Staging path not found: {staging_path}")
        print("   Run migration without --dry-run first\n")
        return

    # Get list of person folders
    folders = [
        f for f in os.listdir(staging_path)
        if os.path.isdir(os.path.join(staging_path, f))
    ]

    print(f"Found {len(folders)} person folders in {staging_path}\n")

    with app.app_context():
        added = 0
        skipped = 0

        for folder in folders:
            folder_path = os.path.join(staging_path, folder)

            # Count images
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
            ]
            image_count = len(image_files)

            # Check if already in database
            existing = Person.query.filter_by(
                domain='serbia',
                normalized_name=folder
            ).first()

            if existing:
                skipped += 1
                continue

            if dry_run:
                print(f"   🔍 Would add: {folder} ({image_count} images)")
                added += 1
                continue

            # Load name mapping if exists
            mapping_file = os.path.join(staging_path, 'name_mapping.json')
            full_name = folder.replace('_', ' ').title()

            if os.path.exists(mapping_file):
                import json
                try:
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        mappings = json.load(f)
                        full_name = mappings.get(folder, full_name)
                except:
                    pass

            # Create person record
            person = Person(
                domain='serbia',
                full_name=full_name,
                normalized_name=folder,
                folder_path=folder_path,
                total_images=image_count,
                status='completed',  # Already trained
                images_from_wikimedia=0,  # Unknown for existing data
                images_from_serp=image_count  # Assume all from SERP
            )

            db.session.add(person)
            added += 1

            if added % 10 == 0:
                print(f"   Added {added} people...")

        if not dry_run and added > 0:
            db.session.commit()

        print(f"\n📊 Summary:")
        print(f"   Added: {added}")
        print(f"   Skipped (already in DB): {skipped}")
        print(f"   Total: {added + skipped}\n")


def main():
    parser = argparse.ArgumentParser(description='Migrate to multi-domain architecture')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Domain Migration Script")
    print("=" * 60)

    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made\n")

    # Create Flask app
    app = create_app()

    try:
        # Phase 1: Migrate folders
        migrate_folders(dry_run=args.dry_run)

        # Phase 2: Create default domain
        create_default_domain(app, dry_run=args.dry_run)

        # Phase 3: Populate database
        scan_and_populate_database(app, dry_run=args.dry_run)

        print("=" * 60)
        if args.dry_run:
            print("✅ DRY RUN COMPLETE")
            print("\nRun without --dry-run to apply changes:")
            print("  python migrations/migrate_to_multi_domain.py")
        else:
            print("✅ MIGRATION COMPLETE")
            print("\nNext steps:")
            print("  1. Test the application")
            print("  2. Create additional domains: POST /api/domains")
            print("  3. Start using domain-aware endpoints")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
