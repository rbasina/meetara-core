#!/usr/bin/env python3
"""
Domain Mapping Report - Shows detailed mapping between downloads folders and domain_config.yaml
"""

import sys
import os
from pathlib import Path
import yaml

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.config_loader import config_loader

def get_downloads_domains():
    """Get all domain folders from downloads directory"""
    downloads_dir = Path("downloads")
    if not downloads_dir.exists():
        return []
    
    domains = []
    for item in downloads_dir.iterdir():
        if item.is_dir():
            domains.append(item.name)
    return sorted(domains)

def get_config_domains_by_category():
    """Get all domains from config organized by category"""
    config_path = Path("config/domain_config.yaml")
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    domains_by_category = {}
    for category, category_config in config.get('categories', {}).items():
        domains = list(category_config.get('domains', {}).keys())
        domains_by_category[category] = domains
    
    return domains_by_category

def find_domain_in_config(domain_name, domains_by_category):
    """Find which category a domain belongs to"""
    for category, domains in domains_by_category.items():
        if domain_name in domains:
            return category
    return None

def main():
    print("\n" + "="*80)
    print("DOMAIN MAPPING REPORT - Downloads vs Domain Config")
    print("="*80)
    
    # Get domains from downloads folder
    downloads_domains = get_downloads_domains()
    
    # Get domains from config
    config_domains_by_category = get_config_domains_by_category()
    all_config_domains = []
    for domains in config_domains_by_category.values():
        all_config_domains.extend(domains)
    all_config_domains = sorted(set(all_config_domains))
    
    print(f"\nSUMMARY:")
    print(f"   Downloads folders: {len(downloads_domains)}")
    print(f"   Config domains: {len(all_config_domains)}")
    print(f"   Config categories: {len(config_domains_by_category)}")
    
    # Detailed mapping
    print(f"\n" + "="*80)
    print("DETAILED MAPPING: Downloads Folders -> Config")
    print("="*80)
    
    mapped = []
    unmapped = []
    invalid = []
    
    for domain in sorted(downloads_domains):
        category = find_domain_in_config(domain, config_domains_by_category)
        
        if domain.lower() in ['new folder', 'new', 'untitled']:
            invalid.append((domain, "INVALID NAME - Should be renamed"))
            print(f"\n   [{domain}]")
            print(f"      Status: INVALID FOLDER NAME")
            print(f"      Recommendation: Rename to 'senior_health' (files are about aging)")
        elif category:
            mapped.append((domain, category))
            tier = config_loader.get_domain_config(domain).get('tier', 'unknown')
            validation = config_loader.get_domain_config(domain).get('requires_validation', False)
            print(f"\n   [{domain}]")
            print(f"      Status: MAPPED")
            print(f"      Category: {category}")
            print(f"      Tier: {tier}")
            print(f"      Validation Required: {validation}")
        else:
            unmapped.append(domain)
            print(f"\n   [{domain}]")
            print(f"      Status: NOT FOUND IN CONFIG")
            print(f"      Recommendation: Add to domain_config.yaml")
    
    # Show unmapped domains that need to be added
    if unmapped:
        print(f"\n" + "="*80)
        print("UNMAPPED DOMAINS - Need to be added to config")
        print("="*80)
        for domain in unmapped:
            print(f"   - {domain}")
    
    # Show all config categories and domains
    print(f"\n" + "="*80)
    print("ALL CONFIG CATEGORIES AND DOMAINS")
    print("="*80)
    
    for category, domains in sorted(config_domains_by_category.items()):
        tier = "unknown"
        try:
            # Get category tier from config
            config_path = Path("config/domain_config.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            category_config = config.get('categories', {}).get(category, {})
            tier = category_config.get('tier', 'unknown')
        except:
            pass
        
        print(f"\n   [{category}] (Tier: {tier})")
        print(f"      Domains ({len(domains)}):")
        for domain in sorted(domains):
            status = "IN USE" if domain in downloads_domains else "AVAILABLE"
            marker = "[*]" if domain in downloads_domains else "[ ]"
            print(f"         {marker} {domain} ({status})")
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"   Mapped domains: {len(mapped)}")
    print(f"   Unmapped domains: {len(unmapped)}")
    print(f"   Invalid folders: {len(invalid)}")
    print(f"   Total config domains: {len(all_config_domains)}")
    print(f"   Total downloads folders: {len(downloads_domains)}")
    
    if unmapped:
        print(f"\n   RECOMMENDATION: Add {len(unmapped)} missing domain(s) to config")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

