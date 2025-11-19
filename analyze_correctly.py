#!/usr/bin/env python3
"""
CORRECT Database Analysis Script

This script properly extracts person names and analyzes photo counts.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json
import re
from datetime import datetime

class CorrectAnalyzer:
    """Properly analyzes face recognition databases"""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def __init__(self):
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

    def extract_person_name_correct(self, filename: str) -> str:
        """
        CORRECT extraction of person name from filename.

        Pattern analysis:
        - Aca_Lukas_2025-05-29_15000311.png -> Aca_Lukas
        - Abraham_Nnamdi_Nwankwo_2023-04-08_1745863063780.jpg -> Abraham_Nnamdi_Nwankwo
        - Matej_Erjavec_2024-09-04_1745798977931.jpg -> Matej_Erjavec

        The pattern is: PersonName_YYYY-MM-DD_timestamp.ext
        We want to extract everything BEFORE the first YYYY-MM-DD pattern.
        """
        name_part = Path(filename).stem

        # Look for date pattern YYYY-MM-DD or YYYY-XX-XX where X is digit
        # This is more reliable than looking for just a year
        match = re.search(r'_(\d{4}-\d{2}-\d{2})', name_part)
        if match:
            # Everything before the date is the person name
            person_name = name_part[:match.start()]
            return person_name

        # Fallback: look for pattern with year followed by timestamp
        # Like: Name_2025_timestamp or Name_2024_timestamp
        match = re.search(r'_(20\d{2})_\d{10,}', name_part)
        if match:
            person_name = name_part[:match.start()]
            return person_name

        # Another fallback: just year-month-day pattern with underscores
        match = re.search(r'_(20\d{2})_\d{2}_\d{2}_', name_part)
        if match:
            person_name = name_part[:match.start()]
            return person_name

        # If no pattern found, return the whole name (might be test data)
        return name_part

    def analyze_sample_filenames(self, directory: Path, sample_size: int = 100) -> dict:
        """Analyze a sample of filenames to understand the pattern"""
        print(f"\n🔍 Analyzing filename patterns in: {directory}")

        try:
            files = [f for f in directory.iterdir() if f.is_file() and self.is_image_file(f.name)]
            sample = files[:sample_size]

            patterns = []
            for f in sample:
                extracted = self.extract_person_name_correct(f.name)
                patterns.append({
                    'original': f.name,
                    'extracted_name': extracted
                })

            # Show first 10 examples
            print(f"   Sample filename patterns (first 10):")
            for i, p in enumerate(patterns[:10], 1):
                print(f"   {i:2}. {p['original'][:60]:<60} -> '{p['extracted_name']}'")

            return patterns

        except Exception as e:
            print(f"   Error analyzing patterns: {e}")
            return []

    def analyze_directory(self, directory: Path, domain_name: str, storage_type: str) -> dict:
        """Analyze a directory with correct person name extraction"""

        print(f"\n{'='*70}")
        print(f"Analyzing: {storage_type}/{domain_name}")
        print(f"Path: {directory}")
        print(f"{'='*70}")

        result = {
            'domain': domain_name,
            'storage_type': storage_type,
            'path': str(directory),
            'exists': directory.exists(),
            'total_photos': 0,
            'total_persons': 0,
            'persons': {},
            'total_size_bytes': 0,
            'file_types': defaultdict(int),
            'sample_files': []
        }

        if not directory.exists():
            print(f"   ✗ Directory does not exist")
            return result

        # Collect all image files (including from subdirectories if batched)
        all_image_files = []

        # Check if this is a batched structure
        subdirs = [d for d in directory.iterdir() if d.is_dir() and d.name.startswith('batch_')]

        if subdirs:
            print(f"   Found {len(subdirs)} batch subdirectories")
            for subdir in sorted(subdirs):
                batch_files = [f for f in subdir.iterdir() if f.is_file() and self.is_image_file(f.name)]
                all_image_files.extend(batch_files)
                print(f"   - {subdir.name}: {len(batch_files)} files")
        else:
            all_image_files = [f for f in directory.iterdir() if f.is_file() and self.is_image_file(f.name)]
            print(f"   Found {len(all_image_files)} image files (flat structure)")

        # Show filename pattern samples
        if all_image_files:
            print(f"\n   📝 Filename pattern examples:")
            for i, f in enumerate(all_image_files[:5], 1):
                extracted = self.extract_person_name_correct(f.name)
                print(f"      {f.name[:65]:<65} -> '{extracted}'")

        # Process all files
        person_data = defaultdict(lambda: {
            'count': 0,
            'size': 0,
            'files': []
        })

        print(f"\n   Processing {len(all_image_files)} files...")

        for img_file in all_image_files:
            person_name = self.extract_person_name_correct(img_file.name)
            file_size = img_file.stat().st_size
            file_ext = img_file.suffix.lower()

            person_data[person_name]['count'] += 1
            person_data[person_name]['size'] += file_size
            person_data[person_name]['files'].append({
                'filename': img_file.name,
                'size': file_size
            })

            result['total_size_bytes'] += file_size
            result['file_types'][file_ext] += 1

        result['total_photos'] = len(all_image_files)
        result['total_persons'] = len(person_data)

        # Convert person data to final format
        for person_name, data in person_data.items():
            result['persons'][person_name] = {
                'image_count': data['count'],
                'total_size_bytes': data['size'],
                'total_size_human': self.get_file_size_human(data['size']),
                'sample_files': [f['filename'] for f in data['files'][:5]]  # Store first 5 as samples
            }

        # Calculate statistics
        image_counts = [p['image_count'] for p in result['persons'].values()]
        if image_counts:
            result['statistics'] = {
                'avg_photos_per_person': sum(image_counts) / len(image_counts),
                'median_photos_per_person': sorted(image_counts)[len(image_counts)//2],
                'min_photos_per_person': min(image_counts),
                'max_photos_per_person': max(image_counts),
                'total_size_human': self.get_file_size_human(result['total_size_bytes'])
            }

            # Distribution
            distribution = defaultdict(int)
            for count in image_counts:
                if count == 1:
                    distribution['1_photo'] += 1
                elif count <= 5:
                    distribution['2-5_photos'] += 1
                elif count <= 10:
                    distribution['6-10_photos'] += 1
                elif count <= 20:
                    distribution['11-20_photos'] += 1
                elif count <= 50:
                    distribution['21-50_photos'] += 1
                else:
                    distribution['50+_photos'] += 1

            result['distribution'] = dict(distribution)

        # Print summary
        print(f"\n   ✅ Results:")
        print(f"      Total Persons:     {result['total_persons']:,}")
        print(f"      Total Photos:      {result['total_photos']:,}")
        print(f"      Total Size:        {self.get_file_size_human(result['total_size_bytes'])}")

        if 'statistics' in result:
            stats = result['statistics']
            print(f"      Avg Photos/Person: {stats['avg_photos_per_person']:.1f}")
            print(f"      Median:            {stats['median_photos_per_person']}")
            print(f"      Min-Max:           {stats['min_photos_per_person']}-{stats['max_photos_per_person']}")

            print(f"\n   📊 Distribution:")
            total = result['total_persons']
            for key, count in sorted(result['distribution'].items()):
                pct = (count / total * 100) if total > 0 else 0
                print(f"      {key:<15} {count:>6,} ({pct:>5.1f}%)")

        # Show top persons
        top_persons = sorted(result['persons'].items(),
                           key=lambda x: x[1]['image_count'],
                           reverse=True)[:20]

        if top_persons:
            print(f"\n   🏆 Top 20 Persons by Photo Count:")
            print(f"      {'Rank':<6} {'Person Name':<40} {'Photos':<10} {'Size'}")
            print(f"      {'-'*80}")
            for i, (name, data) in enumerate(top_persons, 1):
                display_name = name[:37] + '...' if len(name) > 40 else name
                print(f"      {i:<6} {display_name:<40} {data['image_count']:<10} {data['total_size_human']}")

        return result

    def analyze_all_storages(self):
        """Analyze all storage locations correctly"""

        print(f"\n{'#'*70}")
        print(f"# CORRECT DATABASE ANALYSIS")
        print(f"# Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}\n")

        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'storages': {}
        }

        storage_locations = {
            'prod': 'storage/recognized_faces_prod',
            'batched': 'storage/recognized_faces_batched',
            'test': 'storage/recognized_faces_test'
        }

        for storage_type, base_path in storage_locations.items():
            base = Path(base_path)

            if not base.exists():
                continue

            storage_result = {
                'base_path': base_path,
                'domains': {}
            }

            # Find all domain directories
            domain_dirs = [d for d in base.iterdir() if d.is_dir()]

            for domain_dir in sorted(domain_dirs):
                domain_result = self.analyze_directory(domain_dir, domain_dir.name, storage_type)
                storage_result['domains'][domain_dir.name] = domain_result

            results['storages'][storage_type] = storage_result

        return results

    def print_final_summary(self, results: dict):
        """Print final comprehensive summary"""

        print(f"\n\n{'#'*70}")
        print(f"# FINAL SUMMARY")
        print(f"{'#'*70}\n")

        # Aggregate across all storages and domains
        all_persons = {}
        grand_total_photos = 0
        grand_total_size = 0

        for storage_type, storage_data in results['storages'].items():
            for domain_name, domain_data in storage_data['domains'].items():
                print(f"\n📁 {storage_type.upper()}/{domain_name}:")
                print(f"   Persons: {domain_data['total_persons']:,}")
                print(f"   Photos:  {domain_data['total_photos']:,}")
                print(f"   Size:    {self.get_file_size_human(domain_data['total_size_bytes'])}")

                if 'statistics' in domain_data:
                    stats = domain_data['statistics']
                    print(f"   Avg/Person: {stats['avg_photos_per_person']:.1f}")

                grand_total_photos += domain_data['total_photos']
                grand_total_size += domain_data['total_size_bytes']

                # Track persons across storages
                for person_name, person_data in domain_data['persons'].items():
                    key = f"{storage_type}:{domain_name}:{person_name}"
                    all_persons[key] = {
                        'name': person_name,
                        'storage': storage_type,
                        'domain': domain_name,
                        'photo_count': person_data['image_count'],
                        'size': person_data['total_size_bytes']
                    }

        print(f"\n{'='*70}")
        print(f"📊 GRAND TOTALS:")
        print(f"   Total Photos:  {grand_total_photos:,}")
        print(f"   Total Size:    {self.get_file_size_human(grand_total_size)}")
        print(f"   Total Entries: {len(all_persons):,} (person-storage combinations)")
        print(f"{'='*70}\n")

    def save_json(self, results: dict, filename: str = 'correct_analysis.json'):
        """Save results to JSON"""
        try:
            # Clean up defaultdict before saving
            def clean_for_json(obj):
                if isinstance(obj, defaultdict):
                    return dict(obj)
                elif isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                else:
                    return obj

            cleaned_results = clean_for_json(results)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON report saved to: {filename}")
            print(f"   File size: {Path(filename).stat().st_size / 1024:.1f} KB")

        except Exception as e:
            print(f"❌ Failed to save JSON: {e}")


def main():
    """Main execution"""

    if not Path('app').exists():
        print("❌ Please run from project root")
        sys.exit(1)

    analyzer = CorrectAnalyzer()

    # Run analysis
    results = analyzer.analyze_all_storages()

    # Print summary
    analyzer.print_final_summary(results)

    # Save JSON
    analyzer.save_json(results)

    print(f"\n{'='*70}")
    print("✅ Analysis Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
