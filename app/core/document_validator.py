"""
Document Quality Validator for Meetara Core
Assesses document quality, relevance, and content before vector storage
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from loguru import logger

class DocumentValidator:
    """Validates document quality and relevance before vector storage."""
    
    def __init__(self):
        self.quality_threshold = 0.2  # Much more lenient for all document types
        self.relevance_threshold = 0.1  # Much more lenient for all document types
        
    def validate_document(self, content: str, filename: str, domain: str) -> Dict[str, any]:
        """Comprehensive document validation."""
        try:
            validation_result = {
                "is_valid": True,
                "quality_score": 0.0,
                "relevance_score": 0.0,
                "issues": [],
                "recommendations": []
            }
            
            # 1. Basic quality checks
            quality_score = self._assess_quality(content, filename)
            validation_result["quality_score"] = quality_score
            
            # 2. Domain relevance check
            relevance_score = self._assess_domain_relevance(content, domain)
            validation_result["relevance_score"] = relevance_score
            
            # 3. Content structure check
            structure_issues = self._check_content_structure(content)
            validation_result["issues"].extend(structure_issues)
            
            # 4. Determine if document should be accepted
            if quality_score < self.quality_threshold:
                validation_result["is_valid"] = False
                validation_result["recommendations"].append("Document quality too low")
            
            if relevance_score < self.relevance_threshold:
                validation_result["is_valid"] = False
                validation_result["recommendations"].append("Document not relevant to domain")
            
            if len(structure_issues) > 3:
                validation_result["is_valid"] = False
                validation_result["recommendations"].append("Too many structural issues")
            
            logger.info(f"Document validation: {filename} - Quality: {quality_score:.2f}, Relevance: {relevance_score:.2f}, Valid: {validation_result['is_valid']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in document validation: {e}")
            return {
                "is_valid": False,
                "quality_score": 0.0,
                "relevance_score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": ["Document validation failed"]
            }
    
    def _assess_quality(self, content: str, filename: str) -> float:
        """Assess document quality based on various metrics (generalized approach)."""
        try:
            score = 0.0
            total_checks = 0
            
            # 1. Content length check
            if len(content.strip()) > 50:
                score += 1.0
            elif len(content.strip()) > 20:
                score += 0.5
            total_checks += 1
            
            # 2. Text structure check
            sentences = content.split('.')
            if len(sentences) > 2:
                score += 1.0
            elif len(sentences) > 1:
                score += 0.5
            total_checks += 1
            
            # 3. Readability check
            words = content.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            if 2 <= avg_word_length <= 12:
                score += 1.0
            elif 1 <= avg_word_length <= 15:
                score += 0.5
            total_checks += 1
            
            # 4. Special characters and formatting
            if not re.search(r'[^\w\s\.\,\!\?\;\:\-\(\)]', content):
                score += 1.0  # Clean text
            else:
                score += 0.5  # Accept documents with special characters
            total_checks += 1
            
            # 5. Information density
            unique_words = len(set(words))
            if len(words) > 0:
                diversity_ratio = unique_words / len(words)
                if diversity_ratio > 0.6:
                    score += 1.0
                elif diversity_ratio > 0.4:
                    score += 0.5
            total_checks += 1
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return 0.0
    
    def _assess_domain_relevance(self, content: str, domain: str) -> float:
        """Assess how relevant the document is to the specified domain using config-driven keywords."""
        try:
            content_lower = content.lower()
            score = 0.0
            
            # Get domain keywords from config (generalized approach)
            domain_keywords = self._get_domain_keywords(domain)
            
            if domain_keywords:
                matches = sum(1 for keyword in domain_keywords if keyword.lower() in content_lower)
                if matches > 0:
                    score = min(matches / len(domain_keywords), 1.0)
            
            # Give minimum score for any content to be more lenient
            if len(content.strip()) > 0:
                score = max(score, 0.1)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in relevance assessment: {e}")
            return 0.1  # Return minimum score instead of 0
    
    def _get_domain_keywords(self, domain: str) -> List[str]:
        """Get keywords for domain relevance checking."""
        try:
            from app.core.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            return config_loader.get_domain_keywords(domain)
        except Exception as e:
            logger.error(f"Error getting domain keywords: {e}")
            return []
    
    def _check_content_structure(self, content: str) -> List[str]:
        """Check for structural issues in the document."""
        issues = []
        
        try:
            # Check for empty content
            if not content.strip():
                issues.append("Empty content")
            
            # Check for excessive whitespace
            if len(content) > 0 and len(content.strip()) / len(content) < 0.8:
                issues.append("Excessive whitespace")
            
            # Check for repetitive content
            words = content.split()
            if len(words) > 10:
                unique_words = len(set(words))
                if unique_words / len(words) < 0.3:
                    issues.append("Repetitive content")
            
            # Check for very short content
            if len(content.strip()) < 50:
                issues.append("Content too short")
            
            # Check for encoding issues
            if 'â' in content or 'â' in content:
                issues.append("Possible encoding issues")
            
        except Exception as e:
            issues.append(f"Structure check error: {str(e)}")
        
        return issues
    
    def batch_validate_documents(self, documents: List[Dict]) -> Dict[str, List[Dict]]:
        """Validate multiple documents and categorize results."""
        try:
            results = {
                "accepted": [],
                "rejected": [],
                "needs_review": []
            }
            
            for doc in documents:
                validation = self.validate_document(
                    doc.get("content", ""),
                    doc.get("filename", ""),
                    doc.get("domain", "")
                )
                
                if validation["is_valid"]:
                    results["accepted"].append({
                        "document": doc,
                        "validation": validation
                    })
                elif validation["quality_score"] > 0.3 or validation["relevance_score"] > 0.3:
                    results["needs_review"].append({
                        "document": doc,
                        "validation": validation
                    })
                else:
                    results["rejected"].append({
                        "document": doc,
                        "validation": validation
                    })
            
            logger.info(f"Batch validation complete: {len(results['accepted'])} accepted, {len(results['rejected'])} rejected, {len(results['needs_review'])} need review")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch validation: {e}")
            return {"accepted": [], "rejected": [], "needs_review": []}

# Global instance
document_validator = DocumentValidator() 