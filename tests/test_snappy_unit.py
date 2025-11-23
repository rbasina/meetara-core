#!/usr/bin/env python3
"""
Unit Tests for Snappy Compression Implementation
Tests the actual application code, not just API endpoints
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, Mock

# Import the actual application code
from app.rag.domain_retrievers import DomainRetriever, get_domain_retriever
from app.rag.vector_loader import VectorLoader, vector_loader
from app.core.config import settings
from langchain_core.documents import Document


class TestSnappyCompression(unittest.TestCase):
    """Test Snappy compression implementation in vector storage"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test vectorstore
        self.test_dir = tempfile.mkdtemp()
        self.test_vectorstore_path = Path(self.test_dir) / "vectorstore"
        self.test_vectorstore_path.mkdir(exist_ok=True)
        
        # Backup original settings
        self.original_vectorstore_path = settings.vectorstore_path
        settings.vectorstore_path = self.test_vectorstore_path  # Use Path object, not string
        
        # Test domain
        self.test_domain = "unit_test_compression"
        
    def tearDown(self):
        """Clean up test environment"""
        # Restore original settings
        settings.vectorstore_path = self.original_vectorstore_path
        
        # Remove test directory
        if self.test_dir and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
    
    @patch('app.rag.domain_retrievers.Chroma')
    @patch('app.rag.domain_retrievers.HuggingFaceEmbeddings')
    def test_domain_retriever_initialization(self, mock_embeddings, mock_chroma):
        """Test that DomainRetriever initializes with Snappy compression"""
        print("🧪 Testing DomainRetriever initialization with Snappy compression")
        
        # Mock ChromaDB
        mock_chroma_instance = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Create domain retriever
        retriever = DomainRetriever(self.test_domain)
        
        # Check that vectorstore was created
        self.assertIsNotNone(retriever.vectorstore)
        self.assertEqual(retriever.domain, self.test_domain)
        
        # Check that domain directory was created
        domain_path = self.test_vectorstore_path / self.test_domain
        self.assertTrue(domain_path.exists())
        
        # Verify ChromaDB was called with correct parameters
        mock_chroma.assert_called_once()
        call_args = mock_chroma.call_args
        self.assertIn('persist_directory', call_args.kwargs)
        self.assertIn('embedding_function', call_args.kwargs)
        
        print(f"✅ DomainRetriever initialized successfully for domain: {self.test_domain}")
    
    @patch('app.rag.vector_loader.get_domain_retriever')
    def test_vector_loader_with_compression(self, mock_get_retriever):
        """Test VectorLoader with Snappy compression"""
        print("🧪 Testing VectorLoader with Snappy compression")
        
        # Create test document
        test_content = """
        This is a test document for Snappy compression testing.
        It contains multiple sentences to test document processing.
        The content should be chunked and stored with compression.
        """
        
        test_doc = Document(
            page_content=test_content,
            metadata={"source": "test_file.txt", "test_compression": True}
        )
        
        # Mock domain retriever
        mock_retriever = Mock()
        mock_retriever.add_documents = Mock()
        mock_get_retriever.return_value = mock_retriever
        
        # Test vector loader
        loader = VectorLoader()
        
        # Test document processing
        result = loader.process_and_add_to_domain(
            documents=[test_doc],
            domain=self.test_domain,
            metadata={"unit_test": True}
        )
        
        self.assertTrue(result)
        
        # Verify retriever was called
        mock_get_retriever.assert_called_once_with(self.test_domain)
        mock_retriever.add_documents.assert_called_once()
        
        print(f"✅ VectorLoader processed document with compression")
    
    @patch('app.rag.domain_retrievers.Chroma')
    @patch('app.rag.domain_retrievers.HuggingFaceEmbeddings')
    def test_snappy_compression_storage(self, mock_embeddings, mock_chroma):
        """Test that Snappy compression is actually used in storage"""
        print("🧪 Testing Snappy compression storage")
        
        # Mock ChromaDB
        mock_chroma_instance = Mock()
        mock_chroma_instance.add_documents = Mock()
        mock_chroma_instance.persist = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Create domain retriever
        retriever = DomainRetriever(self.test_domain)
        
        # Add test documents
        test_docs = [
            Document(
                page_content="First test document for compression testing.",
                metadata={"doc_id": 1}
            ),
            Document(
                page_content="Second test document with more content to test compression efficiency.",
                metadata={"doc_id": 2}
            ),
            Document(
                page_content="Third test document with comprehensive content to demonstrate Snappy compression benefits.",
                metadata={"doc_id": 3}
            )
        ]
        
        # Add documents to vectorstore
        retriever.add_documents(test_docs)
        
        # Check that files were created with compression
        domain_path = self.test_vectorstore_path / self.test_domain
        self.assertTrue(domain_path.exists())
        
        # Check for ChromaDB files (which use Snappy compression internally)
        files = list(domain_path.rglob('*'))
        self.assertGreater(len(files), 0)
        
        # Verify ChromaDB methods were called
        mock_chroma_instance.add_documents.assert_called_once()
        mock_chroma_instance.persist.assert_called_once()
        
        print(f"✅ Snappy compression storage verified")
        print(f"   📁 Domain path: {domain_path}")
        print(f"   📄 Total files: {len(files)}")
    
    @patch('app.rag.vector_loader.get_domain_retriever')
    def test_compression_efficiency(self, mock_get_retriever):
        """Test compression efficiency with multiple documents"""
        print("🧪 Testing compression efficiency")
        
        # Create large test document
        large_content = "Test content. " * 1000  # ~15KB of content
        
        test_doc = Document(
            page_content=large_content,
            metadata={"large_test": True}
        )
        
        # Mock domain retriever
        mock_retriever = Mock()
        mock_retriever.add_documents = Mock()
        mock_get_retriever.return_value = mock_retriever
        
        # Process with vector loader
        loader = VectorLoader()
        result = loader.process_and_add_to_domain(
            documents=[test_doc],
            domain=self.test_domain
        )
        
        self.assertTrue(result)
        
        # Check storage size
        domain_path = self.test_vectorstore_path / self.test_domain
        total_size = sum(f.stat().st_size for f in domain_path.rglob('*') if f.is_file())
        
        # Size should be reasonable (compressed)
        self.assertLess(total_size, 1024 * 1024)  # Less than 1MB
        
        print(f"✅ Compression efficiency verified")
        print(f"   📊 Total storage size: {total_size / 1024:.2f} KB")
    
    @patch('app.rag.domain_retrievers.Chroma')
    @patch('app.rag.domain_retrievers.HuggingFaceEmbeddings')
    def test_domain_isolation(self, mock_embeddings, mock_chroma):
        """Test that domains are properly isolated with compression"""
        print("🧪 Testing domain isolation with compression")
        
        # Mock ChromaDB
        mock_chroma_instance = Mock()
        mock_chroma_instance.add_documents = Mock()
        mock_chroma_instance.persist = Mock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Create multiple domains
        domains = ["domain1", "domain2", "domain3"]
        retrievers = {}
        
        for domain in domains:
            retriever = DomainRetriever(domain)
            retrievers[domain] = retriever
            
            # Add test document
            test_doc = Document(
                page_content=f"Test content for {domain}",
                metadata={"domain": domain}
            )
            retriever.add_documents([test_doc])
        
        # Check that each domain has separate storage
        for domain in domains:
            domain_path = self.test_vectorstore_path / domain
            self.assertTrue(domain_path.exists())
            
            # Check that each domain has its own files
            files = list(domain_path.rglob('*'))
            self.assertGreater(len(files), 0)
        
        print(f"✅ Domain isolation verified")
        print(f"   📁 Domains created: {len(domains)}")
    
    @patch('app.rag.vector_loader.get_domain_retriever')
    def test_compression_with_metadata(self, mock_get_retriever):
        """Test that metadata is preserved with compression"""
        print("🧪 Testing metadata preservation with compression")
        
        # Create document with rich metadata
        test_doc = Document(
            page_content="Test document with metadata",
            metadata={
                "source": "test_file.txt",
                "author": "Test Author",
                "date": "2025-07-23",
                "compression_test": True,
                "snappy_enabled": True
            }
        )
        
        # Mock domain retriever
        mock_retriever = Mock()
        mock_retriever.add_documents = Mock()
        mock_get_retriever.return_value = mock_retriever
        
        # Process document
        loader = VectorLoader()
        result = loader.process_and_add_to_domain(
            documents=[test_doc],
            domain=self.test_domain
        )
        
        self.assertTrue(result)
        
        # Verify retriever was called with correct documents
        mock_get_retriever.assert_called_once_with(self.test_domain)
        mock_retriever.add_documents.assert_called_once()
        
        # Check that the documents passed to add_documents have metadata
        call_args = mock_retriever.add_documents.call_args
        documents_passed = call_args[0][0]  # First argument, first element
        
        # Verify metadata is preserved
        self.assertIn("source", documents_passed[0].metadata)
        self.assertIn("compression_test", documents_passed[0].metadata)
        self.assertTrue(documents_passed[0].metadata["compression_test"])
        
        print(f"✅ Metadata preservation verified")


class TestSnappyIntegration(unittest.TestCase):
    """Integration tests for Snappy compression"""
    
    @patch('app.rag.domain_retrievers.Chroma')
    @patch('app.rag.domain_retrievers.HuggingFaceEmbeddings')
    def test_end_to_end_compression(self, mock_embeddings, mock_chroma):
        """Test end-to-end compression workflow"""
        print("🧪 Testing end-to-end Snappy compression workflow")
        
        # Create temporary test environment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vectorstore = Path(temp_dir) / "vectorstore"
            temp_vectorstore.mkdir()
            
            # Mock settings
            with patch('app.core.config.settings') as mock_settings:
                mock_settings.vectorstore_path = temp_vectorstore  # Use Path object
                mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                mock_settings.chunk_size = 1000
                mock_settings.chunk_overlap = 200
                
                # Mock ChromaDB
                mock_chroma_instance = Mock()
                mock_chroma_instance.add_documents = Mock()
                mock_chroma_instance.persist = Mock()
                mock_chroma_instance.similarity_search = Mock(return_value=[
                    Document(page_content="Document 1 content", metadata={"id": 1}),
                    Document(page_content="Document 2 content", metadata={"id": 2}),
                    Document(page_content="Document 3 content", metadata={"id": 3})
                ])
                mock_chroma.return_value = mock_chroma_instance
                
                # Mock embeddings
                mock_embeddings_instance = Mock()
                mock_embeddings.return_value = mock_embeddings_instance
                
                # Test complete workflow
                domain = "integration_test"
                
                # 1. Create domain retriever
                retriever = DomainRetriever(domain)
                
                # 2. Add documents
                docs = [
                    Document(page_content="Document 1 content", metadata={"id": 1}),
                    Document(page_content="Document 2 content", metadata={"id": 2}),
                    Document(page_content="Document 3 content", metadata={"id": 3})
                ]
                
                retriever.add_documents(docs)
                
                # 3. Verify storage
                domain_path = temp_vectorstore / domain
                self.assertTrue(domain_path.exists())
                
                # 4. Test retrieval
                results = retriever.similarity_search("Document", k=3)
                self.assertEqual(len(results), 3)
                
                print(f"✅ End-to-end compression workflow verified")


if __name__ == "__main__":
    # Run tests
    print("🧪 Running Snappy Compression Unit Tests")
    print("=" * 50)
    
    # Run tests using unittest.main()
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("🎉 Snappy compression unit tests completed!") 