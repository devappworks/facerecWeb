#!/usr/bin/env python3
"""
Complete Storage Analysis Script for Face Recognition System

This script analyzes ALL storage locations:
1. storage/recognized_faces_prod/ (production - flat structure)
2. storage/recognized_faces_batched/ (batched structure)
3. storage/recognized_faces/ (staging/original)
4. storage/recognized_faces_test/ (testing)

Usage:
    python analyze_all_storages.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import json
from datetime import datetime

class CompleteStorageAnalyzer:
    """Analyzes all face recognition storage locations"""

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    def __init__(self):
        self.storage_locations = {
            'prod': 'storage/recognized_faces_prod',
            'batched': 'storage/recognized_faces_batched',
            'staging': 'storage/recognized_faces',
            'test': 'storage/recognized_faces_test'
        }
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
        """Extract person name from filename"""
        name_part = Path(filename).stem
        parts = name_part.split('_')

        name_parts = []
        for i, part in enumerate(parts):
            if re.match(r'^(19|20)\d{2}$', part):
                break
            if re.match(r'^\d{10,}$', part):
                break
            name_parts.append(part)

        person_name = '_'.join(name_parts) if name_parts else name_part
        return person_name

    def analyze_flat_directory(self, domain_path: Path) -> Dict:
        """Analyze a flat file structure (all images in domain root)"""
        result = {
            'type': 'flat',
            'path': str(domain_path),
            'exists': domain_path.exists(),
            'total_photos': 0,
            'total_persons': 0,
            'persons': {},
            'total_size_bytes': 0
        }

        if not domain_path.exists():
            return result

        # Get all image files
        try:
            all_files = [f for f in domain_path.iterdir()
                        if f.is_file() and self.is_image_file(f.name)]
        except Exception as e:
            result['error'] = str(e)
            return result

        person_images = defaultdict(int)
        person_sizes = defaultdict(int)

        for img_file in all_files:
            person_name = self.extract_person_name(img_file.name)
            file_size = img_file.stat().st_size

            person_images[person_name] += 1
            person_sizes[person_name] += file_size
            result['total_size_bytes'] += file_size

        result['total_photos'] = len(all_files)
        result['total_persons'] = len(person_images)

        for person_name, count in person_images.items():
            result['persons'][person_name] = {
                'image_count': count,
                'total_size_bytes': person_sizes[person_name]
            }

        return result

    def analyze_batched_directory(self, domain_path: Path) -> Dict:
        """Analyze batched structure (batch_XXXX subdirectories)"""
        result = {
            'type': 'batched',
            'path': str(domain_path),
            'exists': domain_path.exists(),
            'total_photos': 0,
            'total_persons': 0,
            'persons': {},
            'batches': [],
            'total_size_bytes': 0
        }

        if not domain_path.exists():
            return result

        # Find all batch directories
        try:
            batch_dirs = [d for d in domain_path.iterdir()
                         if d.is_dir() and d.name.startswith('batch_')]
        except Exception as e:
            result['error'] = str(e)
            return result

        person_images = defaultdict(int)
        person_sizes = defaultdict(int)

        for batch_dir in sorted(batch_dirs):
            batch_info = {
                'name': batch_dir.name,
                'image_count': 0,
                'size_bytes': 0
            }

            try:
                image_files = [f for f in batch_dir.iterdir()
                              if f.is_file() and self.is_image_file(f.name)]

                batch_info['image_count'] = len(image_files)

                for img_file in image_files:
                    person_name = self.extract_person_name(img_file.name)
                    file_size = img_file.stat().st_size

                    person_images[person_name] += 1
                    person_sizes[person_name] += file_size
                    batch_info['size_bytes'] += file_size
                    result['total_size_bytes'] += file_size

                result['batches'].append(batch_info)
                result['total_photos'] += batch_info['image_count']

            except Exception as e:
                batch_info['error'] = str(e)
                result['batches'].append(batch_info)

        result['total_persons'] = len(person_images)

        for person_name, count in person_images.items():
            result['persons'][person_name] = {
                'image_count': count,
                'total_size_bytes': person_sizes[person_name]
            }

        return result

    def analyze_storage_location(self, storage_type: str, base_path: str) -> Dict:
        """Analyze a complete storage location"""
        print(f"\n{'='*70}")
        print(f"Analyzing Storage: {storage_type.upper()} ({base_path})")
        print(f"{'='*70}")

        base = Path(base_path)

        result = {
            'storage_type': storage_type,
            'base_path': base_path,
            'exists': base.exists(),
            'domains': {}
        }

        if not base.exists():
            print(f"   ✗ Path does not exist")
            return result

        # Find all domain directories
        try:
            domain_dirs = [d for d in base.iterdir() if d.is_dir()]
        except Exception as e:
            result['error'] = str(e)
            print(f"   ✗ Error reading directory: {e}")
            return result

        print(f"   Found {len(domain_dirs)} domain(s): {', '.join([d.name for d in domain_dirs])}")

        for domain_dir in sorted(domain_dirs):
            domain_name = domain_dir.name

            # Check if this is a batched structure
            has_batches = any(d.is_dir() and d.name.startswith('batch_')
                            for d in domain_dir.iterdir())

            if has_batches:
                domain_result = self.analyze_batched_directory(domain_dir)
            else:
                domain_result = self.analyze_flat_directory(domain_dir)

            result['domains'][domain_name] = domain_result

            print(f"   {domain_name}: {domain_result['total_persons']:,} persons, "
                  f"{domain_result['total_photos']:,} photos, "
                  f"{self.get_file_size_human(domain_result['total_size_bytes'])}")

        return result

    def analyze_all(self) -> Dict:
        """Analyze all storage locations"""
        print(f"\n{'#'*70}")
        print(f"# COMPLETE STORAGE ANALYSIS")
        print(f"# Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*70}")

        results = {}

        for storage_type, base_path in self.storage_locations.items():
            results[storage_type] = self.analyze_storage_location(storage_type, base_path)

        return results

    def print_summary(self, analysis: Dict):
        """Print comprehensive summary"""
        print(f"\n\n{'#'*70}")
        print(f"# COMPREHENSIVE SUMMARY")
        print(f"{'#'*70}\n")

        # Overall totals across all storage locations
        grand_total_photos = 0
        grand_total_size = 0
        all_persons = defaultdict(lambda: {'photos': 0, 'size': 0, 'locations': []})

        for storage_type, storage_data in analysis.items():
            if not storage_data.get('exists'):
                continue

            for domain_name, domain_data in storage_data.get('domains', {}).items():
                grand_total_photos += domain_data.get('total_photos', 0)
                grand_total_size += domain_data.get('total_size_bytes', 0)

                # Track persons across all locations
                for person_name, person_data in domain_data.get('persons', {}).items():
                    all_persons[person_name]['photos'] += person_data['image_count']
                    all_persons[person_name]['size'] += person_data['total_size_bytes']
                    all_persons[person_name]['locations'].append(f"{storage_type}:{domain_name}")

        print(f"📊 Grand Totals Across All Storage:")
        print(f"   Total Unique Persons:  {len(all_persons):,}")
        print(f"   Total Photos:          {grand_total_photos:,}")
        print(f"   Total Storage Size:    {self.get_file_size_human(grand_total_size)}")
        if len(all_persons) > 0:
            print(f"   Avg Photos/Person:     {grand_total_photos/len(all_persons):.1f}")
        print()

        # Per-storage breakdown
        print(f"📁 Per-Storage Location Summary:")
        print(f"   {'Storage':<15} {'Domains':<8} {'Persons':<10} {'Photos':<12} {'Size':<12}")
        print(f"   {'-'*65}")

        for storage_type, storage_data in analysis.items():
            if not storage_data.get('exists'):
                print(f"   {storage_type:<15} {'N/A':<8} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
                continue

            total_domains = len(storage_data.get('domains', {}))
            total_persons = sum(d.get('total_persons', 0)
                              for d in storage_data.get('domains', {}).values())
            total_photos = sum(d.get('total_photos', 0)
                             for d in storage_data.get('domains', {}).values())
            total_size = sum(d.get('total_size_bytes', 0)
                           for d in storage_data.get('domains', {}).values())

            print(f"   {storage_type:<15} {total_domains:<8} {total_persons:<10,} "
                  f"{total_photos:<12,} {self.get_file_size_human(total_size):<12}")

        print()

        # Person statistics across all locations
        print(f"👥 Person Distribution Analysis:")
        sorted_persons = sorted(all_persons.items(),
                               key=lambda x: x[1]['photos'],
                               reverse=True)

        if sorted_persons:
            image_counts = [p[1]['photos'] for p in sorted_persons]
            avg_images = sum(image_counts) / len(image_counts)
            median_images = sorted(image_counts)[len(image_counts)//2]

            print(f"   Total unique persons:  {len(sorted_persons):,}")
            print(f"   Avg images/person:     {avg_images:.1f}")
            print(f"   Median images/person:  {median_images}")
            print(f"   Min images:            {min(image_counts)}")
            print(f"   Max images:            {max(image_counts)}")
            print()

            # Top persons
            print(f"   🏆 Top 20 persons by total images:")
            print(f"      {'Rank':<6} {'Person Name':<35} {'Images':<10} {'Size':<12} {'Locations'}")
            print(f"      {'-'*90}")

            for i, (person_name, data) in enumerate(sorted_persons[:20], 1):
                display_name = person_name[:32] + '...' if len(person_name) > 35 else person_name
                locations_str = ', '.join(data['locations'][:2])
                if len(data['locations']) > 2:
                    locations_str += f" +{len(data['locations'])-2}"
                print(f"      {i:<6} {display_name:<35} {data['photos']:<10,} "
                      f"{self.get_file_size_human(data['size']):<12} {locations_str}")

    def save_json_report(self, analysis: Dict, filename: str = 'complete_storage_analysis.json'):
        """Save complete analysis to JSON"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Complete JSON report saved to: {filename}")
        except Exception as e:
            print(f"\n✗ Failed to save JSON report: {e}")


import re  # Add this import at the top

def main():
    """Main execution"""
    if not Path('app').exists() or not Path('config.py').exists():
        print("\n❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)

    analyzer = CompleteStorageAnalyzer()

    # Run complete analysis
    analysis = analyzer.analyze_all()

    # Print summary
    analyzer.print_summary(analysis)

    # Save report
    analyzer.save_json_report(analysis)

    print(f"\n{'='*70}")
    print("Complete analysis finished!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
