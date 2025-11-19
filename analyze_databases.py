#!/usr/bin/env python3
"""
Database Analysis Script for Face Recognition System

This script analyzes the production face recognition databases for each domain,
providing detailed statistics about photos, persons, and pickle files.

Usage:
    python analyze_databases.py

Output:
    - Summary of all databases
    - Per-domain statistics
    - Per-person image counts
    - Pickle file information
    - Health checks and recommendations
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json

class DatabaseAnalyzer:
    """Analyzes face recognition databases"""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    PICKLE_FILENAME = 'representations_vgg_face.pkl'

    def __init__(self, base_path: str = 'storage/recognized_faces_prod'):
        self.base_path = Path(base_path)
        self.results = {}

    def get_file_size_human(self, size_bytes: int) -> str:
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def is_image_file(self, filename: str) -> bool:
        """Check if file is an image"""
        return Path(filename).suffix.lower() in self.IMAGE_EXTENSIONS

    def analyze_domain(self, domain_path: Path) -> Dict:
        """Analyze a single domain database"""
        domain_name = domain_path.name

        print(f"\n{'='*60}")
        print(f"Analyzing domain: {domain_name.upper()}")
        print(f"{'='*60}")

        result = {
            'domain': domain_name,
            'path': str(domain_path),
            'exists': domain_path.exists(),
            'total_photos': 0,
            'total_persons': 0,
            'persons': {},
            'pickle_file': {
                'exists': False,
                'size_bytes': 0,
                'size_human': 'N/A',
                'path': None
            },
            'warnings': [],
            'health_score': 100
        }

        if not domain_path.exists():
            result['warnings'].append(f"Domain directory does not exist: {domain_path}")
            result['health_score'] = 0
            return result

        # Check pickle file
        pickle_path = domain_path / self.PICKLE_FILENAME
        if pickle_path.exists():
            result['pickle_file']['exists'] = True
            result['pickle_file']['size_bytes'] = pickle_path.stat().st_size
            result['pickle_file']['size_human'] = self.get_file_size_human(pickle_path.stat().st_size)
            result['pickle_file']['path'] = str(pickle_path)
            print(f"✓ Pickle file found: {result['pickle_file']['size_human']}")
        else:
            result['warnings'].append(f"Pickle file not found: {pickle_path}")
            result['health_score'] -= 30
            print(f"✗ Pickle file NOT FOUND")

        # Scan all subdirectories (each subdirectory = one person)
        person_dirs = [d for d in domain_path.iterdir() if d.is_dir()]
        result['total_persons'] = len(person_dirs)

        print(f"\nScanning {len(person_dirs)} person directories...")

        for person_dir in person_dirs:
            person_name = person_dir.name

            # Count images in person directory
            image_files = [f for f in person_dir.iterdir()
                          if f.is_file() and self.is_image_file(f.name)]

            image_count = len(image_files)
            result['total_photos'] += image_count

            # Store per-person statistics
            result['persons'][person_name] = {
                'image_count': image_count,
                'path': str(person_dir)
            }

            # Health checks
            if image_count == 0:
                result['warnings'].append(f"Empty person directory: {person_name}")
                result['health_score'] -= 1
            elif image_count < 5:
                result['warnings'].append(f"Low image count for {person_name}: {image_count} images")

        print(f"✓ Found {result['total_persons']} persons")
        print(f"✓ Found {result['total_photos']} total photos")

        return result

    def analyze_all_domains(self) -> Dict:
        """Analyze all domains in the base path"""
        print(f"\n{'#'*60}")
        print(f"# FACE RECOGNITION DATABASE ANALYSIS")
        print(f"# Base path: {self.base_path}")
        print(f"{'#'*60}")

        if not self.base_path.exists():
            print(f"\n❌ ERROR: Base path does not exist: {self.base_path}")
            print(f"   Make sure you're running this from the project root directory.")
            return {'error': 'Base path not found', 'domains': []}

        # Find all domain directories
        domain_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]

        if not domain_dirs:
            print(f"\n⚠️  WARNING: No domain directories found in {self.base_path}")
            return {'domains': []}

        print(f"\nFound {len(domain_dirs)} domain(s): {', '.join([d.name for d in domain_dirs])}")

        # Analyze each domain
        results = []
        for domain_dir in sorted(domain_dirs):
            domain_result = self.analyze_domain(domain_dir)
            results.append(domain_result)

        return {'domains': results}

    def print_summary(self, analysis: Dict):
        """Print summary of all domains"""
        if 'error' in analysis:
            return

        domains = analysis['domains']

        if not domains:
            print("\n⚠️  No domains to analyze")
            return

        print(f"\n\n{'#'*60}")
        print(f"# SUMMARY")
        print(f"{'#'*60}\n")

        # Overall statistics
        total_photos = sum(d['total_photos'] for d in domains)
        total_persons = sum(d['total_persons'] for d in domains)
        total_domains = len(domains)

        print(f"📊 Overall Statistics:")
        print(f"   Total Domains:  {total_domains}")
        print(f"   Total Persons:  {total_persons}")
        print(f"   Total Photos:   {total_photos}")
        print()

        # Per-domain summary
        print(f"📁 Per-Domain Summary:")
        print(f"   {'Domain':<15} {'Persons':<10} {'Photos':<10} {'Pickle':<10} {'Health':<10}")
        print(f"   {'-'*60}")

        for domain in domains:
            pickle_status = '✓' if domain['pickle_file']['exists'] else '✗'
            health = f"{domain['health_score']}%"
            print(f"   {domain['domain']:<15} {domain['total_persons']:<10} "
                  f"{domain['total_photos']:<10} {pickle_status:<10} {health:<10}")

        print()

    def print_detailed_person_stats(self, analysis: Dict, domain_name: str = None, top_n: int = 20):
        """Print detailed per-person statistics"""
        if 'error' in analysis:
            return

        domains = analysis['domains']

        if domain_name:
            domains = [d for d in domains if d['domain'] == domain_name]
            if not domains:
                print(f"\n⚠️  Domain '{domain_name}' not found")
                return

        for domain in domains:
            print(f"\n{'='*60}")
            print(f"Per-Person Statistics: {domain['domain'].upper()}")
            print(f"{'='*60}\n")

            if not domain['persons']:
                print("   No persons found")
                continue

            # Sort persons by image count (descending)
            sorted_persons = sorted(domain['persons'].items(),
                                   key=lambda x: x[1]['image_count'],
                                   reverse=True)

            # Calculate statistics
            image_counts = [p[1]['image_count'] for p in sorted_persons]
            avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
            min_images = min(image_counts) if image_counts else 0
            max_images = max(image_counts) if image_counts else 0

            print(f"   Total persons: {len(sorted_persons)}")
            print(f"   Avg images per person: {avg_images:.1f}")
            print(f"   Min images: {min_images}")
            print(f"   Max images: {max_images}")
            print()

            # Show top N persons
            print(f"   Top {min(top_n, len(sorted_persons))} persons by image count:")
            print(f"   {'Rank':<6} {'Person Name':<35} {'Images':<10}")
            print(f"   {'-'*60}")

            for i, (person_name, person_data) in enumerate(sorted_persons[:top_n], 1):
                # Truncate long names
                display_name = person_name[:32] + '...' if len(person_name) > 35 else person_name
                print(f"   {i:<6} {display_name:<35} {person_data['image_count']:<10}")

            if len(sorted_persons) > top_n:
                remaining = len(sorted_persons) - top_n
                remaining_images = sum(p[1]['image_count'] for p in sorted_persons[top_n:])
                print(f"   ... and {remaining} more persons ({remaining_images} images)")

    def print_warnings(self, analysis: Dict):
        """Print all warnings and recommendations"""
        if 'error' in analysis:
            return

        domains = analysis['domains']

        all_warnings = []
        for domain in domains:
            for warning in domain['warnings']:
                all_warnings.append(f"[{domain['domain']}] {warning}")

        if all_warnings:
            print(f"\n{'='*60}")
            print(f"⚠️  WARNINGS & ISSUES ({len(all_warnings)})")
            print(f"{'='*60}\n")

            for warning in all_warnings:
                print(f"   • {warning}")
            print()

        # Recommendations
        print(f"\n{'='*60}")
        print(f"💡 RECOMMENDATIONS")
        print(f"{'='*60}\n")

        for domain in domains:
            if not domain['pickle_file']['exists']:
                print(f"   • {domain['domain']}: Create pickle file by running face recognition once")

            low_image_persons = [name for name, data in domain['persons'].items()
                                if data['image_count'] < 10]
            if low_image_persons:
                print(f"   • {domain['domain']}: {len(low_image_persons)} persons have <10 images "
                      f"(may reduce recognition accuracy)")

            if domain['total_persons'] == 0:
                print(f"   • {domain['domain']}: Empty database - add training data")

    def save_json_report(self, analysis: Dict, filename: str = 'database_analysis_report.json'):
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Detailed JSON report saved to: {filename}")
        except Exception as e:
            print(f"\n✗ Failed to save JSON report: {e}")


def main():
    """Main execution function"""

    # Check if we're in the right directory
    if not Path('app').exists() or not Path('config.py').exists():
        print("\n❌ ERROR: Please run this script from the project root directory")
        print("   Example: python analyze_databases.py")
        sys.exit(1)

    # Create analyzer
    analyzer = DatabaseAnalyzer()

    # Run analysis
    analysis = analyzer.analyze_all_domains()

    # Print results
    analyzer.print_summary(analysis)

    # Print detailed per-person stats for each domain
    if 'domains' in analysis and analysis['domains']:
        for domain in analysis['domains']:
            analyzer.print_detailed_person_stats(analysis, domain['domain'], top_n=30)

    # Print warnings and recommendations
    analyzer.print_warnings(analysis)

    # Save JSON report
    analyzer.save_json_report(analysis)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
