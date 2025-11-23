#!/usr/bin/env python3
"""
Automated Document Downloader for Meetara Core
Downloads authorized documents from approved sources
"""

import requests
import os
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import logging

class DocumentDownloader:
    """Download documents from authorized sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Authorized sources by domain with real document URLs
        self.sources = {
            "general_health": {
                "pubmed": {
                    "base_url": "https://pubmed.ncbi.nlm.nih.gov/",
                    "search_terms": ["diabetes management", "hypertension treatment", "cancer prevention"],
                    "file_types": [".pdf", ".txt"],
                    "debug": True
                },
                "who": {
                    "base_url": "https://www.who.int/publications",
                    "categories": ["guidelines", "reports", "factsheets"],
                    "file_types": [".pdf", ".docx"],
                    "debug": True
                },
                "cdc": {
                    "base_url": "https://www.cdc.gov/publications/",
                    "categories": ["disease", "prevention", "education"],
                    "file_types": [".pdf", ".html"],
                    "debug": True
                }
            },
            "programming": {
                "github_docs": {
                    "base_url": "https://docs.github.com/",
                    "categories": ["guides", "api", "tutorials"],
                    "file_types": [".md", ".html"],
                    "debug": True
                },
                "mdn": {
                    "base_url": "https://developer.mozilla.org/",
                    "categories": ["web", "javascript", "python"],
                    "file_types": [".html", ".md"],
                    "debug": True
                },
                "python_docs": {
                    "base_url": "https://docs.python.org/",
                    "categories": ["tutorial", "reference", "library"],
                    "file_types": [".html", ".txt"],
                    "debug": True
                }
            },
            "nutrition": {
                "usda": {
                    "base_url": "https://fdc.nal.usda.gov/",
                    "categories": ["nutrition", "dietary", "food"],
                    "file_types": [".pdf", ".csv"],
                    "debug": True
                },
                "who_nutrition": {
                    "base_url": "https://www.who.int/health-topics/nutrition",
                    "categories": ["guidelines", "research"],
                    "file_types": [".pdf", ".docx"],
                    "debug": True
                }
            },
            "academic_tutoring": {
                "openstax": {
                    "base_url": "https://openstax.org",
                    "categories": ["textbooks", "math", "science", "humanities"],
                    "file_types": [".pdf"],
                    "debug": True
                },
                "mit_ocw": {
                    "base_url": "https://ocw.mit.edu",
                    "categories": ["courses", "lectures", "notes"],
                    "file_types": [".pdf", ".html"],
                    "debug": True
                }
            }
        }
    
    def download_domain_documents(self, domain: str, max_documents: int = 50):
        """Download documents for a specific domain"""
        print(f"Downloading documents for domain: {domain}")
        
        if domain not in self.sources:
            print(f"[ERROR] Domain {domain} not found in authorized sources")
            return []
        
        domain_sources = self.sources[domain]
        downloaded_files = []
        
        for source_name, source_config in domain_sources.items():
            print(f"Searching {source_name} for {domain} documents...")
            
            try:
                files = self.download_from_source(source_name, source_config, max_documents)
                downloaded_files.extend(files)
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Error downloading from {source_name}: {e}")
                continue
        
        print(f"Downloaded {len(downloaded_files)} files for {domain}")
        return downloaded_files
    
    def download_from_source(self, source_name: str, source_config: dict, max_docs: int):
        """Download documents from a specific source"""
        files = []
        base_url = source_config["base_url"]
        debug = source_config.get("debug", False)
        
        # Get search terms or categories
        search_terms = source_config.get("search_terms", [])
        categories = source_config.get("categories", [])
        
        if debug:
            print(f"   🔍 Debug: Searching {base_url}")
            print(f"   🔍 Debug: Search terms: {search_terms}")
            print(f"   🔍 Debug: Categories: {categories}")
        
        # Search for documents
        for term in search_terms[:max_docs]:
            try:
                # Search the source
                search_url = f"{base_url}?term={term}"
                if debug:
                    print(f"   🔍 Debug: Searching URL: {search_url}")
                
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                if debug:
                    print(f"   🔍 Debug: Response status: {response.status_code}")
                    print(f"   🔍 Debug: Response length: {len(response.content)} bytes")
                
                # Parse HTML for document links
                soup = BeautifulSoup(response.content, 'html.parser')
                document_links = self.extract_document_links(soup, source_config["file_types"], debug)
                
                if debug:
                    print(f"   🔍 Debug: Found {len(document_links)} document links")
                
                # Download documents
                for link in document_links[:5]:  # Limit per term
                    file_path = self.download_document(link, source_name, term)
                    if file_path:
                        files.append({
                            "path": str(file_path),
                            "source": source_name,
                            "term": term,
                            "url": link
                        })
                
            except Exception as e:
                print(f"⚠️ Error searching {term} in {source_name}: {e}")
                if debug:
                    print(f"   🔍 Debug: Full error: {str(e)}")
                continue
        
        return files
    
    def extract_document_links(self, soup, file_types, debug=False):
        """Extract document links from HTML"""
        links = []
        
        # Look for all links
        all_links = soup.find_all('a', href=True)
        if debug:
            print(f"   🔍 Debug: Found {len(all_links)} total links")
        
        for link in all_links:
            href = link['href']
            
            # Check if it's a document link
            for file_type in file_types:
                if file_type in href.lower():
                    links.append(href)
                    if debug:
                        print(f"   🔍 Debug: Found document link: {href}")
                    break
        
        # If no document links found, try alternative approaches
        if not links and debug:
            print(f"   🔍 Debug: No document links found, trying alternative search...")
            
            # Look for any links that might be documents
            for link in all_links[:10]:  # Check first 10 links
                href = link['href']
                text = link.get_text().lower()
                
                # Look for keywords that might indicate documents
                doc_keywords = ['pdf', 'download', 'document', 'guide', 'manual', 'report']
                if any(keyword in text or keyword in href.lower() for keyword in doc_keywords):
                    if debug:
                        print(f"   🔍 Debug: Potential document link: {href} (text: {text})")
                    links.append(href)
        
        return links
    
    def download_document(self, url: str, source: str, term: str):
        """Download a single document"""
        try:
            # Create download directory
            download_dir = Path("downloads") / source
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file first to check content type
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if it's actually a document (not HTML)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                print(f"⚠️ Skipping HTML page: {url}")
                return None
            
            # Check file size (skip very small files that might be HTML)
            if len(response.content) < 1000:
                print(f"⚠️ Skipping small file (likely HTML): {url}")
                return None
            
            # Check if content looks like HTML
            content_start = response.content[:200].decode('utf-8', errors='ignore').lower()
            if '<html' in content_start or '<!doctype' in content_start:
                print(f"⚠️ Skipping HTML content: {url}")
                return None
            
            # Generate filename based on content type
            if 'pdf' in content_type:
                extension = '.pdf'
            elif 'text' in content_type:
                extension = '.txt'
            elif 'word' in content_type or 'docx' in content_type:
                extension = '.docx'
            else:
                extension = '.txt'  # Default
            
            filename = f"{source}_{term}_{int(time.time())}{extension}"
            file_path = download_dir / filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filename}")
            return file_path
            
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return None

def create_sample_documents():
    """Create sample documents for testing when no downloads are available"""
    print("Creating sample documents for testing...")
    
    sample_docs = {
        "pubmed": {
            "diabetes_management.txt": """
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
        "github_docs": {
            "git_basics.txt": """
Git Basics Guide

Git is a distributed version control system for tracking changes in source code.

Basic Commands:
- git init: Initialize a new repository
- git add: Stage changes for commit
- git commit: Commit staged changes
- git push: Upload commits to remote repository
- git pull: Download changes from remote repository

Best Practices:
- Commit frequently with meaningful messages
- Use branches for new features
- Keep commits atomic and focused
- Review code before merging

Common Workflows:
1. Feature Branch Workflow
2. Gitflow Workflow
3. Trunk-Based Development
            """,
            "api_documentation.txt": """
API Documentation Best Practices

Creating good API documentation is crucial for developer adoption.

Key Elements:
- Clear endpoint descriptions
- Request/response examples
- Authentication methods
- Error handling
- Rate limiting information

Documentation Tools:
- Swagger/OpenAPI
- Postman Collections
- GitHub Pages
- ReadTheDocs

Best Practices:
- Keep documentation up to date
- Include code examples
- Provide interactive testing
- Document all error codes
            """
        },
        "academic_tutoring": {
            "study_techniques.txt": """
Study Techniques Guide

Effective study techniques can significantly improve learning outcomes.

Active Learning Strategies:
- Active recall: Test yourself on material
- Spaced repetition: Review at increasing intervals
- Elaboration: Connect new info to existing knowledge
- Interleaving: Mix different topics/techniques
- Dual coding: Combine words and visuals

Memory Techniques:
- Mnemonics: Use memory aids and acronyms
- Method of loci: Associate info with locations
- Visual imagery: Create mental pictures
- Chunking: Group information into meaningful units

Time Management:
- Pomodoro Technique: 25-minute focused sessions
- Create a study schedule
- Prioritize difficult subjects
- Take regular breaks
- Avoid multitasking

Note-Taking Methods:
- Cornell Method: Structured note-taking
- Mind Mapping: Visual organization
- Outline Method: Hierarchical structure
- Charting: Tabular format

Test Preparation:
- Practice with sample questions
- Teach the material to someone else
- Create study guides
- Form study groups
- Review past exams
            """,
            "learning_styles.txt": """
Learning Styles and Adaptations

Different people learn differently. Understanding your learning style can enhance academic performance.

Visual Learners:
- Learn through seeing and images
- Benefit from diagrams, charts, and graphs
- Use color coding and highlighting
- Create visual summaries and mind maps
- Watch educational videos

Auditory Learners:
- Learn through hearing and listening
- Benefit from lectures and discussions
- Use verbal repetition and mnemonics
- Read aloud or use text-to-speech
- Participate in study groups

Kinesthetic Learners:
- Learn through doing and movement
- Benefit from hands-on activities
- Use physical activities while studying
- Create models or demonstrations
- Take frequent breaks to move

Reading/Writing Learners:
- Learn through reading and writing
- Benefit from textbooks and notes
- Write summaries and explanations
- Create lists and outlines
- Take detailed notes

Universal Strategies:
- Combine multiple learning styles
- Adapt teaching materials to your style
- Use technology to support learning
- Seek feedback and adjust approach
- Maintain motivation and persistence
            """,
            "effective_notetaking.txt": """
Effective Note-Taking Strategies

Good notes are essential for academic success.

Cornell Method:
1. Divide page into three sections
2. Take notes in main section during class
3. Write cues/questions in left margin after class
4. Write summary at bottom
5. Review by covering notes and answering cues

Outline Method:
- Use hierarchical structure with Roman numerals
- Main topics use capital letters
- Subtopics use numbers
- Details use lowercase letters
- Works well for organized lectures

Mapping Method:
- Start with main topic in center
- Branch out with subtopics
- Use keywords and phrases
- Create connections between ideas
- Good for visual learners

Charting Method:
- Create tables with columns and rows
- Compare and contrast information
- Fill in details during class
- Effective for structured information
- Use for chronological or categorical data

Better Note-Taking Tips:
- Come prepared: Review materials before class
- Listen actively: Focus on main ideas
- Use abbreviations: Save time
- Leave spaces: Add details later
- Review quickly: Within 24 hours
- Organize notes: Keep them in order
- Ask questions: Clarify confusing points
- Be selective: Don't write everything
            """
        },
        "usda": {
            "nutrition_guidelines.txt": """
USDA Nutrition Guidelines

The USDA provides dietary guidelines for Americans.

Key Recommendations:
- Make half your plate fruits and vegetables
- Choose whole grains
- Include lean protein sources
- Limit added sugars and sodium
- Stay hydrated with water

MyPlate Guidelines:
- Fruits: 2 cups daily
- Vegetables: 2.5 cups daily
- Grains: 6-8 ounces daily
- Protein: 5-6.5 ounces daily
- Dairy: 3 cups daily

Special Considerations:
- Age-specific recommendations
- Activity level adjustments
- Health condition modifications
            """,
            "food_safety.txt": """
Food Safety Guidelines

Proper food handling prevents foodborne illness.

Key Principles:
- Clean: Wash hands and surfaces
- Separate: Prevent cross-contamination
- Cook: Use proper temperatures
- Chill: Refrigerate promptly

Temperature Guidelines:
- Poultry: 165°F
- Ground meat: 160°F
- Fish: 145°F
- Leftovers: 165°F

Storage Times:
- Refrigerator: 3-4 days
- Freezer: 3-4 months
- Pantry: Check expiration dates
            """
        }
    }
    
    # Create directories and files
    for source, documents in sample_docs.items():
        source_dir = Path("downloads") / source
        source_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in documents.items():
            file_path = source_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"✅ Created sample document: {file_path}")

def main():
    """Main download function"""
    downloader = DocumentDownloader()
    
    # Download for each domain
    domains = ["general_health", "programming", "nutrition", "academic_tutoring"]
    
    for domain in domains:
        print(f"\n{'='*50}")
        print(f"Downloading documents for {domain}")
        print(f"{'='*50}")
        
        files = downloader.download_domain_documents(domain, max_documents=20)
        
        if files:
            print(f"\nDownload Summary for {domain}:")
            print(f"   Files downloaded: {len(files)}")
            print(f"   Location: downloads/{domain}/")
            
            # Save download log
            log_file = Path("downloads") / f"{domain}_download_log.json"
            with open(log_file, 'w') as f:
                json.dump(files, f, indent=2)
            
            print(f"   Log saved: {log_file}")
        else:
            print(f"\nNo documents downloaded for {domain}")
            print("   This is normal - most websites don't have direct document links")
            print("   Creating sample documents for testing...")
            create_sample_documents()
            break  # Only create samples once

if __name__ == "__main__":
    main() 