#!/usr/bin/env python3
"""
Embedding Model Demonstration
Shows how embedding models work with practical examples
"""

import yaml
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_domain_keywords():
    """Load domain keywords for demonstration"""
    with open('config/domain_keywords.yaml', 'r') as f:
        return yaml.safe_load(f)

def demonstrate_embeddings():
    """Demonstrate how embedding models work"""
    print("🧠 Embedding Model Demonstration")
    print("=" * 50)
    
    # Load the embedding model (same as your system)
    print("\n1️⃣ Loading Embedding Model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"✅ Model loaded: all-MiniLM-L6-v2")
    print(f"📏 Vector dimensions: {model.get_sentence_embedding_dimension()}")
    
    # Example 1: Basic text embedding
    print("\n2️⃣ Basic Text Embedding Example:")
    texts = [
        "I have diabetes symptoms",
        "What are the signs of high blood sugar?",
        "How to manage diabetes?",
        "I love pizza and ice cream"
    ]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    print(f"📊 Generated {len(embeddings)} embeddings")
    print(f"📏 Each embedding has {len(embeddings[0])} dimensions")
    
    # Show similarity between related concepts
    print("\n3️⃣ Semantic Similarity Test:")
    similarities = cosine_similarity([embeddings[0]], embeddings[1:])
    
    for i, text in enumerate(texts[1:], 1):
        similarity = similarities[0][i-1]
        print(f"   '{texts[0]}' vs '{text}' = {similarity:.3f}")
    
    # Example 2: Domain keyword matching
    print("\n4️⃣ Domain Keyword Matching:")
    domain_keywords = load_domain_keywords()
    
    # Get healthcare keywords
    health_keywords = domain_keywords['domains']['general_health']['keywords'][:10]
    print(f"🏥 Testing with {len(health_keywords)} healthcare keywords")
    
    # Test query
    test_query = "I have chest pain and shortness of breath"
    query_embedding = model.encode([test_query])
    
    # Find most similar keywords
    keyword_embeddings = model.encode(health_keywords)
    similarities = cosine_similarity(query_embedding, keyword_embeddings)[0]
    
    # Show top 5 matches
    top_matches = sorted(zip(health_keywords, similarities), 
                        key=lambda x: x[1], reverse=True)[:5]
    
    print(f"\n🔍 Query: '{test_query}'")
    print("📋 Top 5 matching keywords:")
    for keyword, similarity in top_matches:
        print(f"   • {keyword}: {similarity:.3f}")
    
    # Example 3: Different ways to say the same thing
    print("\n5️⃣ Semantic Understanding Test:")
    medical_queries = [
        "I have a headache",
        "My head hurts",
        "I'm experiencing cranial pain",
        "I have a migraine",
        "What's the weather like today?"
    ]
    
    medical_embeddings = model.encode(medical_queries)
    similarities = cosine_similarity([medical_embeddings[0]], medical_embeddings[1:])
    
    print(f"\n🔍 Query: '{medical_queries[0]}'")
    print("📋 Similarity to other queries:")
    for i, query in enumerate(medical_queries[1:], 1):
        similarity = similarities[0][i-1]
        print(f"   • '{query}': {similarity:.3f}")
    
    # Example 4: Vector visualization
    print("\n6️⃣ Vector Space Visualization:")
    print("📊 Each text becomes a point in 384-dimensional space")
    print("📏 Similar meanings = closer points")
    print("📐 Distance = semantic difference")
    
    # Show vector values for a simple example
    simple_texts = ["health", "medical", "doctor", "pizza"]
    simple_embeddings = model.encode(simple_texts)
    
    print(f"\n📈 Sample vector values (first 10 dimensions):")
    for text, embedding in zip(simple_texts, simple_embeddings):
        vector_preview = embedding[:10]
        print(f"   '{text}': {vector_preview}")
    
    print("\n✅ Embedding demonstration complete!")
    print("\n💡 Key Insights:")
    print("   • Embeddings capture semantic meaning")
    print("   • Similar concepts have similar vectors")
    print("   • Your RAG system uses this for smart search")
    print("   • 384 dimensions capture rich semantic information")

if __name__ == "__main__":
    try:
        demonstrate_embeddings()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have sentence-transformers installed:")
        print("   pip install sentence-transformers") 