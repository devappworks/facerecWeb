#!/usr/bin/env python3
"""
CLI komanda za migraciju postojećih slika u batch strukturu.

Ova komanda čita postojeće slike iz storage/recognized_faces_prod/{domain}
i organizuje ih u batch-eve od po 5000 slika za optimizovanu pretragu.

Usage:
    python scripts/batch_migration_command.py --domain example.com --dry-run
    python scripts/batch_migration_command.py --domain example.com
    python scripts/batch_migration_command.py --all-domains --dry-run
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Dodaj root direktorijum u Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.batch_management_service import BatchManagementService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BatchMigrationCommand:
    """CLI komanda za batch migraciju slika"""
    
    SOURCE_BASE_PATH = 'storage/recognized_faces_prod'
    
    def __init__(self):
        self.parser = self.create_argument_parser()
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Kreira argument parser za CLI komandu"""
        parser = argparse.ArgumentParser(
            description='Migriraj postojeće slike u batch strukturu za optimizovanu pretragu',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Primeri korišćenja:

  # Dry run za jedan domain (samo provera, bez kopiranja)
  python scripts/batch_migration_command.py --domain example.com --dry-run

  # Migracija za jedan domain
  python scripts/batch_migration_command.py --domain example.com

  # Dry run za sve domain-e
  python scripts/batch_migration_command.py --all-domains --dry-run

  # Migracija za sve domain-e
  python scripts/batch_migration_command.py --all-domains

  # Provera informacija o postojećim batch-evima
  python scripts/batch_migration_command.py --info --domain example.com

  # Lista svih domain-a sa batch strukturom
  python scripts/batch_migration_command.py --list-batch-domains
            """
        )
        
        # Osnovne opcije
        domain_group = parser.add_mutually_exclusive_group(required=True)
        domain_group.add_argument(
            '--domain', 
            type=str, 
            help='Domain za koji se vrši migracija (npr. example.com)'
        )
        domain_group.add_argument(
            '--all-domains', 
            action='store_true', 
            help='Migracija za sve postojeće domain-e'
        )
        
        # Akcije
        action_group = parser.add_mutually_exclusive_group()
        action_group.add_argument(
            '--dry-run', 
            action='store_true', 
            help='Samo prikaži šta bi se uradilo, bez stvarnog kopiranja'
        )
        action_group.add_argument(
            '--info', 
            action='store_true', 
            help='Prikaži informacije o postojećim batch-evima'
        )
        action_group.add_argument(
            '--list-batch-domains', 
            action='store_true', 
            help='Lista svih domain-a koji imaju batch strukturu'
        )
        
        # Dodatne opcije
        parser.add_argument(
            '--force', 
            action='store_true', 
            help='Prepiši postojeće batch-eve ako postoje'
        )
        parser.add_argument(
            '--delete-originals', 
            action='store_true', 
            help='Briši originalne slike nakon kopiranja u batch-eve'
        )
        parser.add_argument(
            '--no-pickle', 
            action='store_true', 
            help='Ne kreiraj pickle fajlove automatski (brže, ali batch recognition neće raditi)'
        )
        parser.add_argument(
            '--verbose', '-v', 
            action='store_true', 
            help='Verbose output sa dodatnim detaljima'
        )
        
        return parser
    
    def get_available_domains(self) -> list:
        """Vraća listu svih dostupnih domain-a u source direktorijumu"""
        if not os.path.exists(self.SOURCE_BASE_PATH):
            logger.warning(f"Source path does not exist: {self.SOURCE_BASE_PATH}")
            return []
        
        domains = []
        try:
            for item in os.listdir(self.SOURCE_BASE_PATH):
                item_path = os.path.join(self.SOURCE_BASE_PATH, item)
                if os.path.isdir(item_path):
                    domains.append(item)
            
            return sorted(domains)
            
        except Exception as e:
            logger.error(f"Error reading source directory: {str(e)}")
            return []
    
    def get_domain_source_path(self, domain: str) -> str:
        """Vraća putanju do source foldera za dati domain"""
        return os.path.join(self.SOURCE_BASE_PATH, domain)
    
    def check_domain_exists(self, domain: str) -> bool:
        """Proverava da li domain folder postoji"""
        source_path = self.get_domain_source_path(domain)
        return os.path.exists(source_path) and os.path.isdir(source_path)
    
    def migrate_single_domain(self, domain: str, dry_run: bool = False, force: bool = False, 
                            delete_originals: bool = False, create_pickle_files: bool = True) -> dict:
        """
        Migrira jedan domain u batch strukturu
        
        Args:
            domain: Domain za migraciju
            dry_run: Samo provera bez kopiranja
            force: Prepiši postojeće batch-eve
            delete_originals: Briši originalne slike
            create_pickle_files: Kreiraj pickle fajlove
            
        Returns:
            Dict sa rezultatima
        """
        logger.info(f"{'='*60}")
        logger.info(f"MIGRATING DOMAIN: {domain}")
        logger.info(f"{'='*60}")
        
        # Proveri da li domain postoji
        if not self.check_domain_exists(domain):
            error_msg = f"Domain folder does not exist: {domain}"
            logger.error(error_msg)
            return {
                "status": "error",
                "domain": domain,
                "message": error_msg
            }
        
        source_path = self.get_domain_source_path(domain)
        
        # Proveri da li već postoji batch struktura
        if not dry_run and not force:
            existing_info = BatchManagementService.get_batch_info(domain)
            if existing_info["status"] == "success":
                warning_msg = f"Batch structure already exists for domain: {domain}. Use --force to overwrite."
                logger.warning(warning_msg)
                return {
                    "status": "warning",
                    "domain": domain,
                    "message": warning_msg,
                    "existing_info": existing_info
                }
        
        # Pokreni migraciju
        logger.info(f"Source path: {source_path}")
        logger.info(f"Delete originals: {delete_originals}")
        logger.info(f"Create pickle files: {create_pickle_files}")
        result = BatchManagementService.create_batches_from_source(
            source_path=source_path,
            domain=domain,
            dry_run=dry_run,
            delete_originals=delete_originals,
            create_pickle_files=create_pickle_files
        )
        
        # Dodaj domain u rezultat
        result["domain"] = domain
        result["source_path"] = source_path
        
        # Log rezultate
        if result["status"] == "success":
            if dry_run:
                logger.info(f"✅ Dry run completed for {domain}")
                details = result.get("details", {})
                logger.info(f"   Total images: {details.get('total_images', 0)}")
                logger.info(f"   Would create {details.get('total_batches', 0)} batches")
            else:
                logger.info(f"✅ Migration completed for {domain}")
                details = result.get("details", {})
                logger.info(f"   Images copied: {details.get('total_images_copied', 0)}")
                logger.info(f"   Batches created: {details.get('total_batches_created', 0)}")
                logger.info(f"   Batch path: {details.get('batch_base_path', 'N/A')}")
        else:
            logger.error(f"❌ Migration failed for {domain}: {result.get('message', 'Unknown error')}")
        
        return result
    
    def migrate_all_domains(self, dry_run: bool = False, force: bool = False, 
                          delete_originals: bool = False, create_pickle_files: bool = True) -> dict:
        """
        Migrira sve dostupne domain-e
        
        Args:
            dry_run: Samo provera bez kopiranja
            force: Prepiši postojeće batch-eve
            delete_originals: Briši originalne slike
            create_pickle_files: Kreiraj pickle fajlove
            
        Returns:
            Dict sa rezultatima za sve domain-e
        """
        logger.info(f"{'='*60}")
        logger.info(f"MIGRATING ALL DOMAINS")
        logger.info(f"{'='*60}")
        
        available_domains = self.get_available_domains()
        
        if not available_domains:
            error_msg = "No domains found in source directory"
            logger.error(error_msg)
            return {
                "status": "error",
                "message": error_msg,
                "source_path": self.SOURCE_BASE_PATH
            }
        
        logger.info(f"Found {len(available_domains)} domains: {', '.join(available_domains)}")
        
        results = {
            "status": "success",
            "total_domains": len(available_domains),
            "domains_processed": 0,
            "domains_succeeded": 0,
            "domains_failed": 0,
            "domains_skipped": 0,
            "results": []
        }
        
        for domain in available_domains:
            try:
                result = self.migrate_single_domain(domain, dry_run, force, delete_originals, create_pickle_files)
                results["results"].append(result)
                results["domains_processed"] += 1
                
                if result["status"] == "success":
                    results["domains_succeeded"] += 1
                elif result["status"] == "warning":
                    results["domains_skipped"] += 1
                else:
                    results["domains_failed"] += 1
                    
            except Exception as e:
                error_result = {
                    "status": "error",
                    "domain": domain,
                    "message": str(e)
                }
                results["results"].append(error_result)
                results["domains_processed"] += 1
                results["domains_failed"] += 1
                logger.error(f"❌ Unexpected error for domain {domain}: {str(e)}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MIGRATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total domains: {results['total_domains']}")
        logger.info(f"Succeeded: {results['domains_succeeded']}")
        logger.info(f"Failed: {results['domains_failed']}")
        logger.info(f"Skipped: {results['domains_skipped']}")
        
        return results
    
    def show_domain_info(self, domain: str) -> dict:
        """Prikazuje informacije o batch strukturi za domain"""
        logger.info(f"{'='*60}")
        logger.info(f"BATCH INFO FOR DOMAIN: {domain}")
        logger.info(f"{'='*60}")
        
        result = BatchManagementService.get_batch_info(domain)
        
        if result["status"] == "not_found":
            logger.info(f"❌ No batch structure found for domain: {domain}")
        else:
            metadata = result.get("metadata", {})
            batches = result.get("existing_batches", [])
            
            logger.info(f"✅ Batch structure found")
            logger.info(f"   Total batches: {metadata.get('total_batches', 0)}")
            logger.info(f"   Images per batch: {metadata.get('images_per_batch', 0)}")
            logger.info(f"   Created: {metadata.get('created', 'N/A')}")
            logger.info(f"   Last updated: {metadata.get('last_updated', 'N/A')}")
            logger.info(f"   Base path: {result.get('batch_base_path', 'N/A')}")
            
            if batches:
                logger.info(f"\n   Batch details:")
                for batch in batches:
                    status = "✅" if batch.get("exists", False) else "❌"
                    actual_count = batch.get("actual_image_count", "?")
                    expected_count = batch.get("image_count", "?")
                    logger.info(f"     {status} {batch.get('batch_id', 'N/A')}: {actual_count}/{expected_count} images")
        
        return result
    
    def list_batch_domains(self) -> list:
        """Lista svih domain-a sa batch strukturom"""
        logger.info(f"{'='*60}")
        logger.info(f"DOMAINS WITH BATCH STRUCTURE")
        logger.info(f"{'='*60}")
        
        domains = BatchManagementService.list_batch_domains()
        
        if not domains:
            logger.info("❌ No domains with batch structure found")
        else:
            logger.info(f"✅ Found {len(domains)} domains with batch structure:")
            for domain in domains:
                logger.info(f"   📁 {domain}")
        
        return domains
    
    def run(self) -> int:
        """Glavna metoda za pokretanje CLI komande"""
        try:
            args = self.parser.parse_args()
            
            # Postavi log level
            if args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.debug("Verbose mode enabled")
            
            logger.info(f"Batch Migration Command started at {datetime.now()}")
            logger.info(f"Arguments: {vars(args)}")
            
            # Izvršavanje na osnovu argumenata
            if args.list_batch_domains:
                self.list_batch_domains()
                return 0
            
            elif args.info:
                if not args.domain:
                    logger.error("--info requires --domain argument")
                    return 1
                self.show_domain_info(args.domain)
                return 0
            
            elif args.all_domains:
                result = self.migrate_all_domains(
                    dry_run=args.dry_run, 
                    force=args.force,
                    delete_originals=args.delete_originals,
                    create_pickle_files=not args.no_pickle
                )
                return 0 if result["domains_failed"] == 0 else 1
            
            elif args.domain:
                result = self.migrate_single_domain(
                    domain=args.domain,
                    dry_run=args.dry_run,
                    force=args.force,
                    delete_originals=args.delete_originals,
                    create_pickle_files=not args.no_pickle
                )
                return 0 if result["status"] in ["success", "warning"] else 1
            
            else:
                logger.error("No valid action specified")
                self.parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return 1
        finally:
            logger.info(f"Batch Migration Command finished at {datetime.now()}")

def main():
    """Entry point za CLI komandu"""
    command = BatchMigrationCommand()
    return command.run()

if __name__ == "__main__":
    sys.exit(main()) 