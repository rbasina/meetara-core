"""
Validator Tool for Meetara Core.

This tool validates responses for hallucination, banned content, and quality issues
before returning them to users.
"""
import re
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.logger import agent_logger


class ValidatorInput(BaseModel):
    """Input schema for validator tool."""
    response: str = Field(description="The response to validate")
    domain: str = Field(description="The domain context for validation")
    query: str = Field(description="The original query")


class ValidatorTool(BaseTool):
    """Tool for validating responses for hallucination and banned content."""
    
    name: str = "validator"
    description: str = "Validate response for hallucination, banned content, and quality issues"
    args_schema: type = ValidatorInput
    
    def _run(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Run the validator tool."""
        try:
            agent_logger.info(f"Validating response for domain: {domain}")
            
            # Perform validation checks
            validation_result = self._perform_validation(response, domain, query)
            
            return {
                "is_valid": validation_result["is_valid"],
                "confidence": validation_result["confidence"],
                "issues": validation_result["issues"],
                "warnings": validation_result["warnings"],
                "recommendations": validation_result["recommendations"]
            }
            
        except Exception as e:
            agent_logger.error(f"Error in validation: {e}")
            return {
                "is_valid": True,  # Default to valid if validation fails
                "confidence": 0.5,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": ["Unable to validate due to technical issues"]
            }
    
    async def _arun(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Async run of the validator tool."""
        return self._run(response, domain, query)
    
    def _perform_validation(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Perform comprehensive validation checks."""
        
        # Initialize result
        result = {
            "is_valid": True,
            "confidence": 0.8,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Perform various validation checks
        hallucination_check = self._check_hallucination(response, query)
        banned_content_check = self._check_banned_content(response)
        quality_check = self._check_response_quality(response, domain)
        
        # Combine results
        result["issues"].extend(hallucination_check["issues"])
        result["issues"].extend(banned_content_check["issues"])
        result["warnings"].extend(quality_check["warnings"])
        
        # Determine overall validity
        result["is_valid"] = len(result["issues"]) == 0
        
        # Calculate confidence
        base_confidence = 0.8
        if hallucination_check["issues"]:
            base_confidence -= 0.3
        if banned_content_check["issues"]:
            base_confidence -= 0.4
        if quality_check["warnings"]:
            base_confidence -= 0.1
        
        result["confidence"] = max(0.1, base_confidence)
        
        # Add recommendations
        if hallucination_check["issues"]:
            result["recommendations"].append("Verify information accuracy")
        if banned_content_check["issues"]:
            result["recommendations"].append("Review content for appropriateness")
        if quality_check["warnings"]:
            result["recommendations"].append("Improve response clarity and completeness")
        
        return result
    
    def _check_hallucination(self, response: str, query: str) -> Dict[str, Any]:
        """Check for potential hallucination in the response."""
        issues = []
        
        # Check for vague or generic responses
        vague_patterns = [
            r"it depends",
            r"it varies",
            r"generally speaking",
            r"in general",
            r"typically",
            r"usually"
        ]
        
        vague_count = 0
        for pattern in vague_patterns:
            if re.search(pattern, response.lower()):
                vague_count += 1
        
        if vague_count > 2:
            issues.append("Response contains too many vague statements")
        
        # Check for contradictory information
        contradictions = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("recommended", "not recommended")
        ]
        
        for pos, neg in contradictions:
            if pos in response.lower() and neg in response.lower():
                issues.append("Response contains contradictory information")
                break
        
        # Check for overly confident claims without evidence
        confident_patterns = [
            r"definitely",
            r"absolutely",
            r"certainly",
            r"without a doubt",
            r"guaranteed"
        ]
        
        confident_count = 0
        for pattern in confident_patterns:
            if re.search(pattern, response.lower()):
                confident_count += 1
        
        if confident_count > 2:
            issues.append("Response contains overly confident claims without evidence")
        
        return {
            "issues": issues,
            "confidence": 0.9 if not issues else 0.6
        }
    
    def _check_banned_content(self, response: str) -> Dict[str, Any]:
        """Check for banned or inappropriate content."""
        issues = []
        
        # Define banned content patterns
        banned_patterns = [
            r"hack.*password",
            r"bypass.*security",
            r"illegal.*activity",
            r"harm.*others",
            r"dangerous.*activity",
            r"exploit.*vulnerability"
        ]
        
        for pattern in banned_patterns:
            if re.search(pattern, response.lower()):
                issues.append(f"Banned content detected: {pattern}")
        
        # Check for inappropriate language
        inappropriate_words = [
            "hate speech",
            "discrimination",
            "violence",
            "harmful"
        ]
        
        for word in inappropriate_words:
            if word in response.lower():
                issues.append(f"Inappropriate content detected: {word}")
        
        return {
            "issues": issues,
            "confidence": 0.95 if not issues else 0.3
        }
    
    def _check_response_quality(self, response: str, domain: str) -> Dict[str, Any]:
        """Check response quality and completeness."""
        warnings = []
        
        # Check response length
        if len(response) < 20:
            warnings.append("Response may be too brief")
        elif len(response) > 1000:
            warnings.append("Response may be too verbose")
        
        # Check for incomplete sentences
        incomplete_patterns = [
            r"[A-Z][^.]*$",  # Ends without period
            r"[A-Z][^!]*$",  # Ends without exclamation
            r"[A-Z][^?]*$"   # Ends without question mark
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, response):
                warnings.append("Response may contain incomplete sentences")
                break
        
        # Check for domain-specific quality issues
        if domain in ["healthcare", "medical", "general_health"]:
            if "consult" not in response.lower() and "doctor" not in response.lower():
                if any(word in response.lower() for word in ["treatment", "medicine", "diagnosis"]):
                    warnings.append("Medical response should include consultation recommendation")
        
        elif domain in ["legal", "legal_assistance"]:
            if "consult" not in response.lower() and "lawyer" not in response.lower():
                if any(word in response.lower() for word in ["legal", "law", "rights"]):
                    warnings.append("Legal response should include professional consultation recommendation")
        
        elif domain in ["financial", "financial_planning"]:
            if "consult" not in response.lower() and "professional" not in response.lower():
                if any(word in response.lower() for word in ["investment", "money", "financial"]):
                    warnings.append("Financial response should include professional consultation recommendation")
        
        # Check for helpfulness
        helpful_indicators = [
            "help",
            "assist",
            "guide",
            "recommend",
            "suggest",
            "provide"
        ]
        
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in response.lower())
        if helpful_count == 0:
            warnings.append("Response may not be sufficiently helpful")
        
        return {
            "warnings": warnings,
            "confidence": 0.8 if not warnings else 0.7
        }
    
    def _check_consistency(self, response: str, query: str) -> Dict[str, Any]:
        """Check for consistency between query and response."""
        issues = []
        
        # Check if response addresses the query
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        
        # Simple keyword overlap check
        overlap = query_keywords.intersection(response_keywords)
        if len(overlap) < len(query_keywords) * 0.3:
            issues.append("Response may not adequately address the query")
        
        return {
            "issues": issues,
            "confidence": 0.9 if not issues else 0.6
        } 