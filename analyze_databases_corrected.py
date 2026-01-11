#!/usr/bin/env python3
"""
Corrected Database Analysis Script for Face Recognition System

This script analyzes the ACTUAL production face recognition databases,
which use a flat file structure with person names in filenames.

Usage:
    python analyze_databases_corrected.py

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
import re
from datetime import datetime

class DatabaseAnalyzer:
    """Analyzes face recognition databases with flat file structure"""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

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

    def extract_person_name(self, filename: str) -> str:
        """
        Extract person name from filename.
        Expected format: PersonName_date_timestamp.jpg
        Example: Abraham_Nnamdi_Nwankwo_2023-04-08_1745863063780.jpg
        """
        # Remove extension
        name_part = Path(filename).stem

        # Try to find the pattern: name followed by date (YYYY-MM-DD or timestamp)
        # Split by underscore and find where the date/timestamp pattern starts
        parts = name_part.split('_')

        # Look for date pattern (YYYY-MM-DD) or timestamp pattern (digits only after date)
        name_parts = []
        for i, part in enumerate(parts):
            # Check if this looks like a year (4 digits starting with 19 or 20)
            if re.match(r'^(19|20)\d{2}$', part):
                # This is likely the start of the date
                break
            # Check if this is a long timestamp (10+ digits)
            if re.match(r'^\d{10,}$', part):
                # This is likely a timestamp
                break
            name_parts.append(part)

        # Join the name parts back together
        person_name = '_'.join(name_parts) if name_parts else name_part

        return person_name

    def analyze_domain(self, domain_path: Path) -> Dict:
        """Analyze a single domain database with flat file structure"""
        domain_name = domain_path.name

        print(f"\n{'='*70}")
        print(f"Analyzing domain: {domain_name.upper()}")
        print(f"{'='*70}")

        result = {
            'domain': domain_name,
            'path': str(domain_path),
            'exists': domain_path.exists(),
            'total_photos': 0,
            'total_persons': 0,
            'persons': {},
            'pickle_files': [],
            'total_pickle_size': 0,
            'warnings': [],
            'health_score': 100,
            'file_types': defaultdict(int),
            'oldest_image': None,
            'newest_image': None,
            'total_size_bytes': 0
        }

        if not domain_path.exists():
            result['warnings'].append(f"Domain directory does not exist: {domain_path}")
            result['health_score'] = 0
            return result

        # Get all files in the domain directory
        try:
            all_files = list(domain_path.iterdir())
        except Exception as e:
            result['warnings'].append(f"Error reading directory: {e}")
            result['health_score'] = 0
            return result

        # Separate pickle files and image files
        pickle_files = [f for f in all_files if f.suffix == '.pkl']
        image_files = [f for f in all_files if f.is_file() and self.is_image_file(f.name)]

        # Analyze pickle files
        print(f"\n📦 Pickle Files:")
        if pickle_files:
            for pkl_file in pickle_files:
                pkl_size = pkl_file.stat().st_size
                result['pickle_files'].append({
                    'name': pkl_file.name,
                    'size_bytes': pkl_size,
                    'size_human': self.get_file_size_human(pkl_size),
                    'path': str(pkl_file)
                })
                result['total_pickle_size'] += pkl_size
                print(f"   ✓ {pkl_file.name}: {self.get_file_size_human(pkl_size)}")
        else:
            result['warnings'].append("No pickle files found")
            result['health_score'] -= 20
            print(f"   ✗ No pickle files found")

        # Analyze images
        print(f"\n📸 Analyzing {len(image_files)} image files...")

        person_images = defaultdict(list)
        oldest_time = None
        newest_time = None

        for img_file in image_files:
            # Extract person name from filename
            person_name = self.extract_person_name(img_file.name)

            # Get file info
            file_stat = img_file.stat()
            file_size = file_stat.st_size
            file_mtime = file_stat.st_mtime

            # Track file type
            file_ext = img_file.suffix.lower()
            result['file_types'][file_ext] += 1

            # Track total size
            result['total_size_bytes'] += file_size

            # Track oldest/newest
            if oldest_time is None or file_mtime < oldest_time:
                oldest_time = file_mtime
                result['oldest_image'] = {
                    'filename': img_file.name,
                    'date': datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }

            if newest_time is None or file_mtime > newest_time:
                newest_time = file_mtime
                result['newest_image'] = {
                    'filename': img_file.name,
                    'date': datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }

            # Add to person's image list
            person_images[person_name].append({
                'filename': img_file.name,
                'size_bytes': file_size,
                'modified_time': file_mtime
            })

        # Calculate per-person statistics
        result['total_photos'] = len(image_files)
        result['total_persons'] = len(person_images)

        for person_name, images in person_images.items():
            result['persons'][person_name] = {
                'image_count': len(images),
                'total_size_bytes': sum(img['size_bytes'] for img in images),
                'total_size_human': self.get_file_size_human(sum(img['size_bytes'] for img in images))
            }

        # Health checks
        if result['total_persons'] == 0:
            result['warnings'].append("No persons found in database")
            result['health_score'] -= 50

        # Check for persons with very few images
        low_image_persons = [name for name, data in result['persons'].items()
                            if data['image_count'] < 5]
        if low_image_persons:
            result['warnings'].append(f"{len(low_image_persons)} persons have fewer than 5 images")
            result['health_score'] -= min(20, len(low_image_persons))

        print(f"\n✓ Found {result['total_persons']:,} unique persons")
        print(f"✓ Found {result['total_photos']:,} total photos")
        print(f"✓ Total size: {self.get_file_size_human(result['total_size_bytes'])}")
        print(f"✓ Pickle files: {len(pickle_files)} ({self.get_file_size_human(result['total_pickle_size'])})")

        return result

    def analyze_all_domains(self) -> Dict:
        """Analyze all domains in the base path"""
        print(f"\n{'#'*70}")
        print(f"# FACE RECOGNITION DATABASE ANALYSIS (CORRECTED)")
        print(f"# Base path: {self.base_path}")
        print(f"# Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

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

        return {
            'domains': results,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def print_summary(self, analysis: Dict):
        """Print summary of all domains"""
        if 'error' in analysis:
            return

        domains = analysis['domains']

        if not domains:
            print("\n⚠️  No domains to analyze")
            return

        print(f"\n\n{'#'*70}")
        print(f"# SUMMARY")
        print(f"{'#'*70}\n")

        # Overall statistics
        total_photos = sum(d['total_photos'] for d in domains)
        total_persons = sum(d['total_persons'] for d in domains)
        total_domains = len(domains)
        total_size = sum(d['total_size_bytes'] for d in domains)
        total_pickle_size = sum(d['total_pickle_size'] for d in domains)

        print(f"📊 Overall Statistics:")
        print(f"   Total Domains:        {total_domains}")
        print(f"   Total Persons:        {total_persons:,}")
        print(f"   Total Photos:         {total_photos:,}")
        print(f"   Total Storage Size:   {self.get_file_size_human(total_size)}")
        print(f"   Total Pickle Size:    {self.get_file_size_human(total_pickle_size)}")
        if total_persons > 0:
            print(f"   Avg Photos/Person:    {total_photos/total_persons:.1f}")
        print()

        # Per-domain summary
        print(f"📁 Per-Domain Summary:")
        print(f"   {'Domain':<20} {'Persons':<10} {'Photos':<12} {'Avg/Person':<12} {'Size':<12} {'Pickles':<8} {'Health':<8}")
        print(f"   {'-'*90}")

        for domain in domains:
            avg_per_person = domain['total_photos'] / domain['total_persons'] if domain['total_persons'] > 0 else 0
            size_human = self.get_file_size_human(domain['total_size_bytes'])
            pickle_count = len(domain['pickle_files'])
            health = f"{domain['health_score']}%"

            print(f"   {domain['domain']:<20} {domain['total_persons']:<10,} "
                  f"{domain['total_photos']:<12,} {avg_per_person:<12.1f} "
                  f"{size_human:<12} {pickle_count:<8} {health:<8}")

        print()

    def print_detailed_person_stats(self, analysis: Dict, domain_name: str = None, top_n: int = 30, bottom_n: int = 10):
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
            print(f"\n{'='*70}")
            print(f"Per-Person Statistics: {domain['domain'].upper()}")
            print(f"{'='*70}\n")

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
            median_images = sorted(image_counts)[len(image_counts)//2] if image_counts else 0

            print(f"   📊 Statistics:")
            print(f"      Total persons:         {len(sorted_persons):,}")
            print(f"      Total images:          {domain['total_photos']:,}")
            print(f"      Avg images/person:     {avg_images:.1f}")
            print(f"      Median images/person:  {median_images}")
            print(f"      Min images:            {min_images}")
            print(f"      Max images:            {max_images}")

            if domain['oldest_image']:
                print(f"      Oldest image:          {domain['oldest_image']['date']}")
            if domain['newest_image']:
                print(f"      Newest image:          {domain['newest_image']['date']}")

            # File type distribution
            if domain['file_types']:
                print(f"\n   📄 File Types:")
                for ext, count in sorted(domain['file_types'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / domain['total_photos'] * 100) if domain['total_photos'] > 0 else 0
                    print(f"      {ext:<6} {count:>6,} ({percentage:>5.1f}%)")

            print()

            # Distribution analysis
            bins = [0, 1, 5, 10, 20, 50, 100, 200, float('inf')]
            bin_labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '101-200', '200+']
            distribution = defaultdict(int)

            for count in image_counts:
                for i in range(len(bins)-1):
                    if bins[i] < count <= bins[i+1]:
                        distribution[bin_labels[i]] += 1
                        break

            print(f"   📈 Image Count Distribution:")
            print(f"      {'Range':<12} {'Persons':<10} {'Percentage':<12} {'Bar'}")
            print(f"      {'-'*60}")
            for label in bin_labels:
                count = distribution[label]
                percentage = (count / len(sorted_persons) * 100) if len(sorted_persons) > 0 else 0
                bar = '█' * int(percentage / 2)
                print(f"      {label:<12} {count:<10,} {percentage:>5.1f}%       {bar}")
            print()

            # Show top N persons
            print(f"   🏆 Top {min(top_n, len(sorted_persons))} persons by image count:")
            print(f"      {'Rank':<6} {'Person Name':<40} {'Images':<10} {'Size':<12}")
            print(f"      {'-'*70}")

            for i, (person_name, person_data) in enumerate(sorted_persons[:top_n], 1):
                # Truncate long names
                display_name = person_name[:37] + '...' if len(person_name) > 40 else person_name
                print(f"      {i:<6} {display_name:<40} {person_data['image_count']:<10,} {person_data['total_size_human']:<12}")

            if len(sorted_persons) > top_n + bottom_n:
                print(f"      ... ({len(sorted_persons) - top_n - bottom_n} more) ...")

            # Show bottom N persons
            if len(sorted_persons) > top_n and bottom_n > 0:
                print(f"\n   ⚠️  Bottom {min(bottom_n, len(sorted_persons))} persons by image count:")
                print(f"      {'Rank':<6} {'Person Name':<40} {'Images':<10} {'Size':<12}")
                print(f"      {'-'*70}")

                for i, (person_name, person_data) in enumerate(sorted_persons[-bottom_n:], len(sorted_persons)-bottom_n+1):
                    display_name = person_name[:37] + '...' if len(person_name) > 40 else person_name
                    print(f"      {i:<6} {display_name:<40} {person_data['image_count']:<10,} {person_data['total_size_human']:<12}")

            print()

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
            print(f"\n{'='*70}")
            print(f"⚠️  WARNINGS & ISSUES ({len(all_warnings)})")
            print(f"{'='*70}\n")

            for warning in all_warnings:
                print(f"   • {warning}")
            print()

        # Recommendations
        print(f"\n{'='*70}")
        print(f"💡 RECOMMENDATIONS")
        print(f"{'='*70}\n")

        has_recommendations = False
        for domain in domains:
            if len(domain['pickle_files']) == 0:
                print(f"   • {domain['domain']}: Generate pickle files by running face recognition training")
                has_recommendations = True

            low_image_persons = [name for name, data in domain['persons'].items()
                                if data['image_count'] < 10]
            if low_image_persons:
                print(f"   • {domain['domain']}: {len(low_image_persons)} persons have <10 images "
                      f"(consider adding more training data)")
                has_recommendations = True

            if domain['total_persons'] == 0:
                print(f"   • {domain['domain']}: Empty database - add training data")
                has_recommendations = True

            # Check for very large datasets that might need optimization
            if domain['total_photos'] > 50000:
                print(f"   • {domain['domain']}: Large dataset ({domain['total_photos']:,} images) - "
                      f"consider database optimization")
                has_recommendations = True

        if not has_recommendations:
            print(f"   ✓ No major issues found. All databases look healthy!")

    def save_json_report(self, analysis: Dict, filename: str = 'database_analysis_corrected.json'):
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
        print("   Example: python analyze_databases_corrected.py")
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
            if domain['total_persons'] > 0:  # Only show domains with data
                analyzer.print_detailed_person_stats(analysis, domain['domain'], top_n=30, bottom_n=10)

    # Print warnings and recommendations
    analyzer.print_warnings(analysis)

    # Save JSON report
    analyzer.save_json_report(analysis)

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
