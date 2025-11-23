#!/usr/bin/env python3
"""
Test Single Document Upload with Snappy Compression
"""

import requests
import json
from pathlib import Path

def test_single_upload():
    """Test uploading a single document to see Snappy compression"""
    print("🧪 Testing Single Document Upload with Snappy Compression")
    print("=" * 60)
    
    # Test domain
    domain = "test_compression"
    
    # Create domain if it doesn't exist
    try:
        response = requests.post(
            "http://localhost:8000/api/upload/domain",
            data={"domain": domain}
        )
        
        if response.status_code == 200:
            print(f"✅ Domain {domain} ready")
        else:
            print(f"❌ Failed to create domain: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error creating domain: {e}")
        return
    
    # Upload a single document
    test_file = Path("test_documents/general_health/diabetes_guide.txt")
    
    if test_file.exists():
        try:
            with open(test_file, 'rb') as f:
                files = {'file': f}
                data = {
                    'domain': domain,
                    'metadata': json.dumps({
                        "test_compression": True,
                        "snappy_enabled": True
                    })
                }
                
                response = requests.post(
                    "http://localhost:8000/api/upload/doc",
                    files=files,
                    data=data
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Document uploaded successfully!")
                print(f"   📄 File: {test_file.name}")
                print(f"   🎯 Domain: {result.get('domain')}")
                print(f"   📊 Documents added: {result.get('documents_added')}")
                
                # Check vectorstore directory
                vectorstore_path = Path("vectorstore") / domain
                if vectorstore_path.exists():
                    print(f"\n📁 Vectorstore directory: {vectorstore_path}")
                    
                    # List files and their sizes
                    for file_path in vectorstore_path.rglob('*'):
                        if file_path.is_file():
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"   📄 {file_path.name}: {size_mb:.3f} MB")
                            
                            # Check if it's a Parquet file
                            if file_path.suffix == '.parquet':
                                print(f"      🗜️ Snappy compressed Parquet file")
                            elif file_path.suffix == '.sqlite3':
                                print(f"      🗄️ SQLite index file")
                
                # Get domain stats
                stats_response = requests.get(f"http://localhost:8000/api/upload/domain/{domain}/stats")
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    print(f"\n📊 Domain Statistics:")
                    print(f"   📄 Document count: {stats.get('stats', {}).get('count', 0)}")
                    print(f"   🧠 Embedding model: {stats.get('stats', {}).get('embedding_model', 'unknown')}")
                    print(f"   📏 Chunk size: {stats.get('stats', {}).get('chunk_size', 0)}")
                
            else:
                print(f"❌ Upload failed: {response.text}")
                
        except Exception as e:
            print(f"❌ Error uploading document: {e}")
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    test_single_upload() 