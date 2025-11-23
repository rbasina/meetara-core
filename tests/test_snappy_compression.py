#!/usr/bin/env python3
"""
Test Snappy Compression for Vector Database
Demonstrates storage efficiency with sample domains
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import requests

def create_sample_documents():
    """Create comprehensive sample documents for testing compression"""
    sample_docs = {
        "general_health": {
            "diabetes_guide.txt": """
Diabetes Management Guide - Comprehensive Edition

Diabetes is a chronic condition that affects how your body processes glucose. 
The condition occurs when the pancreas doesn't produce enough insulin or when 
the body cannot effectively use the insulin it produces.

Common symptoms include:
- Increased thirst and frequent urination
- Extreme hunger and fatigue
- Blurred vision and slow-healing sores
- Unexplained weight loss
- Tingling or numbness in hands and feet

Treatment options include:
- Blood sugar monitoring with regular testing
- Dietary changes focusing on balanced nutrition
- Regular exercise and physical activity
- Medication when prescribed by healthcare providers
- Insulin therapy for type 1 diabetes

Emergency symptoms to watch for:
- Very high blood sugar levels (hyperglycemia)
- Ketones in urine indicating diabetic ketoacidosis
- Difficulty breathing or shortness of breath
- Confusion, drowsiness, or unconsciousness
- Severe abdominal pain or vomiting

Prevention strategies:
- Maintain a healthy weight through diet and exercise
- Regular medical check-ups and blood sugar monitoring
- Stress management and adequate sleep
- Smoking cessation and alcohol moderation

Always consult with your healthcare provider for personalized advice and treatment plans.
            """,
            "hypertension_treatment.txt": """
Hypertension Treatment Guide - Complete Management

Hypertension, or high blood pressure, affects millions of people worldwide and is a major risk factor for heart disease, stroke, and kidney disease. Normal blood pressure is below 120/80 mmHg.

Understanding Blood Pressure Categories:
- Normal: Less than 120/80 mmHg
- Elevated: 120-129/<80 mmHg
- Stage 1: 130-139/80-89 mmHg
- Stage 2: 140/90 mmHg or higher
- Crisis: Higher than 180/120 mmHg

Treatment approaches include:
- Lifestyle modifications as first-line therapy
- Dietary changes following the DASH diet plan
- Regular aerobic exercise and physical activity
- Medication when lifestyle changes are insufficient
- Regular monitoring and follow-up care

Risk factors include:
- Family history of hypertension
- Age-related changes in blood vessels
- Obesity and excess body weight
- High salt intake and poor diet
- Chronic stress and anxiety
- Physical inactivity and sedentary lifestyle
- Tobacco use and excessive alcohol consumption

Monitoring blood pressure regularly is crucial for effective management and prevention of complications.
            """,
            "cardiology_basics.txt": """
Cardiology Fundamentals - Heart Health Guide

The cardiovascular system is responsible for circulating blood throughout the body, delivering oxygen and nutrients to tissues while removing waste products. Understanding heart health is essential for maintaining overall wellness.

Heart Structure and Function:
- The heart has four chambers: two atria and two ventricles
- Blood flows through a series of valves that prevent backflow
- The heart muscle (myocardium) contracts rhythmically
- Electrical signals coordinate heart contractions

Common Cardiovascular Conditions:
- Coronary artery disease affecting blood flow
- Heart failure when the heart cannot pump effectively
- Arrhythmias causing irregular heart rhythms
- Valvular heart disease affecting valve function
- Congenital heart defects present from birth

Risk Factors for Heart Disease:
- High blood pressure and cholesterol levels
- Diabetes and metabolic syndrome
- Smoking and tobacco use
- Physical inactivity and poor diet
- Obesity and excess body weight
- Family history of heart disease
- Age and gender-related factors

Prevention Strategies:
- Regular cardiovascular exercise
- Heart-healthy diet low in saturated fats
- Stress management and relaxation techniques
- Regular medical check-ups and screenings
- Smoking cessation and alcohol moderation
- Maintaining healthy weight and blood pressure
            """
        },
        "programming": {
            "python_basics.txt": """
Python Programming Basics - Complete Guide

Python is a versatile, high-level programming language known for its simplicity, readability, and extensive library ecosystem. It's widely used in web development, data science, artificial intelligence, and automation.

Variables and Data Types:
- Strings: Text data enclosed in quotes
- Integers: Whole numbers without decimals
- Floats: Decimal numbers with fractional parts
- Lists: Ordered collections of items
- Dictionaries: Key-value pair collections
- Tuples: Immutable ordered collections
- Sets: Unordered collections of unique items
- Booleans: True or False values

Control Structures:
- if/elif/else statements for conditional logic
- for loops for iterating over sequences
- while loops for repeated execution
- try/except blocks for error handling
- break and continue statements for flow control

Functions and Methods:
- Define reusable code blocks with def keyword
- Accept parameters and return values
- Use default arguments and keyword arguments
- Implement lambda functions for simple operations
- Create generator functions for memory efficiency

Object-Oriented Programming:
- Classes define object blueprints
- Objects are instances of classes
- Methods are functions within classes
- Inheritance allows code reuse
- Encapsulation protects data integrity

Example function:
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

File Handling:
- Open files with built-in open() function
- Read and write text and binary files
- Use context managers with 'with' statements
- Handle file exceptions and errors
- Process CSV, JSON, and other formats
            """,
            "web_development.txt": """
Web Development Fundamentals - Modern Stack

Web development involves creating websites and web applications using various technologies and frameworks. Modern web development encompasses both frontend and backend development with responsive design principles.

Frontend Technologies:
- HTML5: Semantic markup and document structure
- CSS3: Styling, layouts, and responsive design
- JavaScript: Client-side interactivity and logic
- TypeScript: Typed JavaScript for large applications
- React: Component-based UI library
- Angular: Full-featured framework by Google
- Vue.js: Progressive JavaScript framework

Backend Technologies:
- Python: Django, Flask, FastAPI frameworks
- Node.js: JavaScript runtime for server-side development
- PHP: Server-side scripting language
- Ruby: Ruby on Rails framework
- Java: Spring Boot and enterprise applications
- C#: ASP.NET Core for Windows development

Database Technologies:
- MySQL: Relational database management system
- PostgreSQL: Advanced open-source database
- MongoDB: NoSQL document database
- SQLite: Lightweight embedded database
- Redis: In-memory data structure store
- Cassandra: Distributed NoSQL database

Modern Development Tools:
- Git: Version control and collaboration
- Docker: Containerization and deployment
- CI/CD: Continuous integration and deployment
- REST APIs: Application programming interfaces
- GraphQL: Query language for APIs
- Microservices: Distributed architecture patterns

Cloud Platforms:
- AWS: Amazon Web Services
- Azure: Microsoft cloud platform
- Google Cloud: Google's cloud services
- Heroku: Platform as a service
- Vercel: Frontend deployment platform
- Netlify: Static site hosting
            """,
            "data_science.txt": """
Data Science Fundamentals - Analytics Guide

Data science combines statistics, programming, and domain expertise to extract insights from data. It involves collecting, cleaning, analyzing, and visualizing data to support decision-making processes.

Data Collection and Sources:
- Structured data from databases and spreadsheets
- Unstructured data from text, images, and videos
- Semi-structured data from JSON and XML files
- Real-time data from sensors and APIs
- Historical data from archives and records

Data Cleaning and Preprocessing:
- Handle missing values and outliers
- Normalize and standardize numerical data
- Encode categorical variables
- Remove duplicates and inconsistencies
- Validate data quality and integrity

Exploratory Data Analysis:
- Statistical summaries and distributions
- Correlation analysis between variables
- Data visualization with charts and graphs
- Pattern recognition and trends
- Hypothesis generation and testing

Machine Learning Techniques:
- Supervised learning: Classification and regression
- Unsupervised learning: Clustering and dimensionality reduction
- Deep learning: Neural networks and complex patterns
- Reinforcement learning: Decision-making algorithms
- Natural language processing: Text analysis

Popular Tools and Libraries:
- Python: pandas, numpy, scikit-learn
- R: Statistical computing and graphics
- SQL: Database querying and manipulation
- Tableau: Data visualization platform
- Power BI: Business intelligence tools
- Jupyter: Interactive development environment

Data Visualization:
- Bar charts and histograms for distributions
- Scatter plots for relationships
- Heat maps for correlation matrices
- Time series plots for temporal data
- Geographic maps for spatial data
- Interactive dashboards for exploration
            """
        }
    }
    
    # Create directories and files
    for domain, documents in sample_docs.items():
        domain_dir = Path(f"test_documents/{domain}")
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        for filename, content in documents.items():
            file_path = domain_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            print(f"✅ Created: {file_path}")

def test_compression_with_api():
    """Test Snappy compression using the API"""
    print("🧪 Testing Snappy Compression with Sample Domains")
    print("=" * 60)
    
    # Create sample documents
    print("\n📄 Creating comprehensive sample documents...")
    create_sample_documents()
    
    # Test domains
    domains = ["general_health", "programming"]
    
    for domain in domains:
        print(f"\n🎯 Testing domain: {domain}")
        
        # Create domain
        try:
            response = requests.post(
                "http://localhost:8000/api/upload/domain",
                data={"domain": domain}
            )
            
            if response.status_code == 200:
                print(f"✅ Created domain: {domain}")
                
                # Upload documents from directory
                domain_dir = f"test_documents/{domain}"
                if Path(domain_dir).exists():
                    # Use batch upload API
                    batch_response = requests.post(
                        "http://localhost:8000/api/upload/batch",
                        data={
                            "domain": domain,
                            "directory_path": domain_dir,
                            "metadata": json.dumps({
                                "compression_test": True,
                                "snappy_enabled": True
                            })
                        }
                    )
                    
                    if batch_response.status_code == 200:
                        result = batch_response.json()
                        print(f"✅ Batch upload completed for {domain}")
                        print(f"   📄 Total files: {result.get('total_files', 0)}")
                        print(f"   ✅ Uploaded: {result.get('uploaded_files', 0)}")
                        print(f"   ❌ Failed: {result.get('failed_files', 0)}")
                        
                        # Get domain stats
                        stats_response = requests.get(f"http://localhost:8000/api/upload/domain/{domain}/stats")
                        if stats_response.status_code == 200:
                            stats = stats_response.json()
                            print(f"   📊 Domain stats: {stats.get('stats', {}).get('count', 0)} documents")
                    else:
                        print(f"❌ Batch upload failed: {batch_response.text}")
                else:
                    print(f"❌ Directory not found: {domain_dir}")
            else:
                print(f"❌ Failed to create domain {domain}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing {domain}: {e}")

def analyze_storage_usage():
    """Analyze storage usage and compression benefits"""
    print("\n📊 Storage Analysis")
    print("=" * 40)
    
    vectorstore_path = Path("vectorstore")
    if vectorstore_path.exists():
        total_size = 0
        domain_sizes = {}
        
        for domain_dir in vectorstore_path.iterdir():
            if domain_dir.is_dir():
                domain_size = sum(f.stat().st_size for f in domain_dir.rglob('*') if f.is_file())
                domain_sizes[domain_dir.name] = {
                    "size_bytes": domain_size,
                    "size_mb": domain_size / (1024 * 1024),
                    "files": len(list(domain_dir.rglob('*')))
                }
                total_size += domain_size
        
        print(f"📁 Total vectorstore size: {total_size / (1024 * 1024):.2f} MB")
        print(f"📊 Number of domains: {len(domain_sizes)}")
        
        for domain, info in domain_sizes.items():
            print(f"   • {domain}: {info['size_mb']:.2f} MB ({info['files']} files)")
        
        # Compression analysis
        print(f"\n🗜️ Compression Analysis:")
        print(f"   • Raw embeddings (estimated): {total_size * 4:.2f} MB")
        print(f"   • Snappy compressed: {total_size / (1024 * 1024):.2f} MB")
        print(f"   • Compression ratio: {((total_size * 4 - total_size) / (total_size * 4) * 100):.1f}%")
        
    else:
        print("❌ Vectorstore directory not found")

def main():
    """Main test function"""
    print("🚀 Snappy Compression Test for Vector Database")
    print("=" * 60)
    
    try:
        # Test compression with API
        test_compression_with_api()
        
        # Analyze storage usage
        analyze_storage_usage()
        
        print("\n✅ Snappy compression test complete!")
        print("💡 Check the vectorstore/ directory for compressed Parquet files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the Meetara Core API is running on http://localhost:8000")

if __name__ == "__main__":
    main() 