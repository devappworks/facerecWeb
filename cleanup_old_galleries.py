#!/usr/bin/env python3
"""
Daily cleanup script for training galleries
Removes galleries older than 1 day
Runs daily at 6 AM via cron
"""
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
GALLERIES_BASE_PATH = "/home/facereco/facerecWeb/storage/serp_originals"
MAX_AGE_HOURS = 24  # Delete galleries older than 24 hours
LOG_FILE = "/home/facereco/facerecWeb/storage/logs/gallery_cleanup.log"

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(log_message.strip())

    # Append to log file
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'a') as f:
            f.write(log_message)
    except Exception as e:
        print(f"Error writing to log file: {e}")

def cleanup_old_galleries():
    """Remove galleries older than MAX_AGE_HOURS"""
    if not os.path.exists(GALLERIES_BASE_PATH):
        log(f"Galleries path does not exist: {GALLERIES_BASE_PATH}")
        return

    cutoff_time = datetime.now() - timedelta(hours=MAX_AGE_HOURS)
    total_deleted = 0
    total_size_freed = 0

    log("=" * 60)
    log("Starting gallery cleanup")
    log(f"Deleting galleries older than: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Iterate through domain directories (e.g., serbia/)
        for domain_dir in Path(GALLERIES_BASE_PATH).iterdir():
            if not domain_dir.is_dir():
                continue

            # Iterate through person directories (e.g., Nenad_Jezdic/)
            for person_dir in domain_dir.iterdir():
                if not person_dir.is_dir():
                    continue

                # Iterate through batch directories (e.g., a8e99054/)
                for batch_dir in person_dir.iterdir():
                    if not batch_dir.is_dir():
                        continue

                    try:
                        # Get directory modification time
                        dir_mtime = datetime.fromtimestamp(batch_dir.stat().st_mtime)

                        if dir_mtime < cutoff_time:
                            # Calculate directory size before deletion
                            dir_size = sum(f.stat().st_size for f in batch_dir.rglob('*') if f.is_file())
                            dir_size_mb = dir_size / (1024 * 1024)

                            # Delete the batch directory
                            shutil.rmtree(batch_dir)
                            total_deleted += 1
                            total_size_freed += dir_size

                            log(f"Deleted: {batch_dir.relative_to(GALLERIES_BASE_PATH)} "
                                f"(age: {(datetime.now() - dir_mtime).days} days, "
                                f"size: {dir_size_mb:.2f} MB)")

                    except Exception as e:
                        log(f"Error processing {batch_dir}: {str(e)}")

    except Exception as e:
        log(f"Error during cleanup: {str(e)}")

    total_freed_mb = total_size_freed / (1024 * 1024)
    log(f"Cleanup complete: Deleted {total_deleted} galleries, freed {total_freed_mb:.2f} MB")
    log("=" * 60)

if __name__ == "__main__":
    cleanup_old_galleries()
