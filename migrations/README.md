# Database Migrations

This directory contains migration scripts for the facerecWeb database.

## Available Migrations

### `migrate_to_multi_domain.py` - Multi-Domain Architecture Migration

Migrates the legacy single-domain structure to the new multi-domain architecture.

**What it does:**
1. Moves `storage/trainingPassSerbia` → `storage/trainingPass/serbia`
2. Creates default 'serbia' domain in database
3. Scans existing person folders and populates database
4. Preserves all existing files (non-destructive)

**Usage:**

```bash
# Dry run (see what would change without making changes)
python migrations/migrate_to_multi_domain.py --dry-run

# Actual migration
python migrations/migrate_to_multi_domain.py
```

**Prerequisites:**
- Flask-SQLAlchemy installed (`pip install Flask-SQLAlchemy`)
- Database initialized (happens automatically on first Flask app start)

**Safe to run multiple times:** The script checks for existing data and skips if already migrated.

## Creating New Migrations

When adding new database changes, create a new migration script following this pattern:

```python
#!/usr/bin/env python
"""
Migration script: Description of what this does

Usage:
    python migrations/my_migration.py [--dry-run]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.database import db

def main():
    parser = argparse.ArgumentParser(description='My migration')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    app = create_app()

    with app.app_context():
        # Your migration code here
        pass

if __name__ == '__main__':
    main()
```

## Rollback

Most migrations are designed to be non-destructive. If you need to rollback:

1. **Database:** Use SQLite commands or delete `storage/training.db`
2. **Files:** Manual rollback (migrations preserve originals where possible)

## Migration Order

If multiple migrations exist, run them in chronological order:

1. `migrate_to_multi_domain.py` (first - establishes baseline)
2. Future migrations...

## Testing Migrations

Always test migrations:

```bash
# 1. Backup your data
cp -r storage/ storage_backup/
cp storage/training.db storage/training.db.backup

# 2. Dry run
python migrations/migrate_to_multi_domain.py --dry-run

# 3. Review output

# 4. Run migration
python migrations/migrate_to_multi_domain.py

# 5. Test application
python run.py

# 6. If issues, restore backup
rm -rf storage/
mv storage_backup/ storage/
mv storage/training.db.backup storage/training.db
```
