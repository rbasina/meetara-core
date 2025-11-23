#!/usr/bin/env python3
"""
Batch Upload Client for Meetara Core
Uses existing API endpoints for batch document uploads
"""

import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class MeetaraBatchClient:
    """Client for batch uploading documents using existing API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def create_domain(self, domain: str) -> bool:
        """Create a new domain using existing API"""
        try:
            response = requests.post(
                f"{self.base_url}/upload/domain",
                data={"domain": domain}
            )
            
            if response.status_code == 200:
                print(f"✅ Created domain: {domain}")
                return True
            else:
                print(f"❌ Failed to create domain {domain}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating domain {domain}: {e}")
            return False
    
    def batch_upload_directory(self, directory_path: str, domain: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Upload all documents from a directory using existing batch API"""
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Use existing batch upload endpoint
            data = {
                "domain": domain,
                "directory_path": directory_path,
                "metadata": json.dumps(metadata) if metadata else None
            }
            
            response = requests.post(
                f"{self.base_url}/upload/batch",
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch upload completed for {domain}")
                print(f"   📄 Total files: {result.get('total_files', 0)}")
                print(f"   ✅ Uploaded: {result.get('uploaded_files', 0)}")
                print(f"   ❌ Failed: {result.get('failed_files', 0)}")
                
                if result.get('errors'):
                    print(f"   ⚠️  Errors: {len(result['errors'])}")
                    for error in result['errors'][:3]:  # Show first 3 errors
                        print(f"      - {error}")
                
                return result
            else:
                print(f"❌ Batch upload failed: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"❌ Error in batch upload: {e}")
            return {"success": False, "error": str(e)}
    
    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get domain statistics using existing API"""
        try:
            response = requests.get(f"{self.base_url}/upload/domain/{domain}/stats")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to get stats for {domain}: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Error getting stats for {domain}: {e}")
            return {}
    
    def list_domains(self) -> List[Dict[str, Any]]:
        """List all domains using existing API"""
        try:
            response = requests.get(f"{self.base_url}/upload/domains")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Failed to list domains: {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error listing domains: {e}")
            return []

def create_sample_documents():
    """Create sample documents for testing"""
    sample_docs = {
        "general_health": {
            "diabetes_guide.txt": """
Diabetes Management Guide

Diabetes is a chronic condition that affects how your body processes glucose.
Common symptoms include increased thirst, frequent urination, and fatigue.

Treatment options include:
- Blood sugar monitoring
- Dietary changes
- Regular exercise
- Medication when prescribed

Emergency symptoms to watch for:
- Very high blood sugar
- Ketones in urine
- Difficulty breathing
- Confusion or drowsiness

Always consult with your healthcare provider for personalized advice.
            """,
            "hypertension_treatment.txt": """
Hypertension Treatment Guide

Hypertension, or high blood pressure, affects millions of people worldwide.
Normal blood pressure is below 120/80 mmHg.

Treatment approaches:
- Lifestyle modifications
- Dietary changes (DASH diet)
- Regular exercise
- Medication when needed

Risk factors include:
- Family history
- Age
- Obesity
- High salt intake
- Stress

Monitoring blood pressure regularly is crucial for management.
            """
        },
        "programming": {
            "python_basics.txt": """
Python Programming Basics

Python is a versatile programming language known for its simplicity.
Here are some fundamental concepts:

Variables and Data Types:
- Strings: text data
- Integers: whole numbers
- Floats: decimal numbers
- Lists: ordered collections
- Dictionaries: key-value pairs

Control Structures:
- if/elif/else statements
- for loops
- while loops

Functions:
- Define reusable code blocks
- Accept parameters
- Return values

Example function:
def greet(name):
    return f"Hello, {name}!"
            """,
            "web_development.txt": """
Web Development Fundamentals

Web development involves creating websites and web applications.
Key technologies include:

Frontend:
- HTML: Structure
- CSS: Styling
- JavaScript: Interactivity

Backend:
- Python (Django, Flask)
- Node.js
- PHP
- Ruby

Databases:
- MySQL
- PostgreSQL
- MongoDB
- SQLite

Modern frameworks:
- React
- Angular
- Vue.js
- Django
- Express.js
            """
        }
    }
    
    # Create directories and files
    for domain, documents in sample_docs.items():
        domain_dir = Path(f"sample_documents/{domain}")
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in documents.items():
            file_path = domain_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"✅ Created: {file_path}")

def main():
    """Main function to demonstrate batch upload using existing API"""
    print("🤖 Meetara Batch Upload Client")
    print("=" * 40)
    
    # Initialize client
    client = MeetaraBatchClient()
    
    # Create sample documents
    print("\n📄 Creating sample documents...")
    create_sample_documents()
    
    # Upload documents for each domain
    domains = ["general_health", "programming"]
    
    for domain in domains:
        print(f"\n🎯 Processing domain: {domain}")
        
        # Create domain
        if client.create_domain(domain):
            # Upload documents from directory
            domain_dir = f"sample_documents/{domain}"
            if Path(domain_dir).exists():
                # Use existing batch API
                result = client.batch_upload_directory(domain_dir, domain)
                
                if result.get('success'):
                    print(f"✅ Batch upload successful for {domain}")
                else:
                    print(f"❌ Batch upload failed for {domain}")
            else:
                print(f"❌ Directory not found: {domain_dir}")
        
        # Get domain stats
        stats = client.get_domain_stats(domain)
        if stats:
            print(f"   📊 Domain stats: {stats.get('stats', {}).get('count', 0)} documents")
    
    # List all domains
    print(f"\n📋 All Domains:")
    domains_list = client.list_domains()
    for domain_info in domains_list:
        print(f"   • {domain_info.get('domain', 'unknown')}: {domain_info.get('document_count', 0)} documents")
    
    print("\n✅ Batch upload client complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the Meetara Core API is running on http://localhost:8000") 