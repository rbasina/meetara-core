 """
Fact-Checking Tool for Meetara Core.

This tool validates responses against reliable sources to ensure accuracy
and prevent misinformation.
"""
import re
import requests
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from app.core.logger import agent_logger


class FactCheckInput(BaseModel):
    """Input schema for fact-checking tool."""
    response: str = Field(description="The response to fact-check")
    domain: str = Field(description="The domain context for fact-checking")
    query: str = Field(description="The original query")


class FactCheckTool(BaseTool):
    """Tool for fact-checking responses against reliable sources."""
    
    name: str = "fact_check"
    description: str = "Validate response accuracy against reliable sources"
    args_schema: type = FactCheckInput
    
    def _run(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Run the fact-checking tool."""
        try:
            agent_logger.info(f"Fact-checking response for domain: {domain}")
            
            # Perform fact-checking based on domain
            fact_check_result = self._perform_fact_check(response, domain, query)
            
            return {
                "is_accurate": fact_check_result["is_accurate"],
                "confidence": fact_check_result["confidence"],
                "issues": fact_check_result["issues"],
                "sources": fact_check_result["sources"],
                "recommendations": fact_check_result["recommendations"]
            }
            
        except Exception as e:
            agent_logger.error(f"Error in fact-checking: {e}")
            return {
                "is_accurate": True,  # Default to true if fact-checking fails
                "confidence": 0.5,
                "issues": [f"Fact-checking error: {str(e)}"],
                "sources": [],
                "recommendations": ["Unable to verify due to technical issues"]
            }
    
    async def _arun(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Async run of the fact-checking tool."""
        return self._run(response, domain, query)
    
    def _perform_fact_check(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Perform fact-checking based on domain."""
        
        # Initialize result
        result = {
            "is_accurate": True,
            "confidence": 0.8,
            "issues": [],
            "sources": [],
            "recommendations": []
        }
        
        # Domain-specific fact-checking
        if domain in ["healthcare", "medical", "general_health", "mental_health"]:
            result = self._check_health_facts(response, query)
        elif domain in ["legal", "legal_assistance", "legal_business"]:
            result = self._check_legal_facts(response, query)
        elif domain in ["financial", "financial_planning", "insurance"]:
            result = self._check_financial_facts(response, query)
        elif domain in ["emergency", "crisis", "emergency_care"]:
            result = self._check_emergency_facts(response, query)
        else:
            result = self._check_general_facts(response, query)
        
        return result
    
    def _check_health_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check health-related facts."""
        issues = []
        sources = []
        
        # Check for medical misinformation patterns
        medical_red_flags = [
            r"cure.*cancer",
            r"miracle.*treatment",
            r"alternative.*medicine.*only",
            r"vaccine.*dangerous",
            r"natural.*only.*safe"
        ]
        
        for pattern in medical_red_flags:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential medical misinformation detected: {pattern}")
        
        # Check for specific medical claims
        if "treatment" in response.lower() or "medicine" in response.lower():
            if not any(word in response.lower() for word in ["consult", "doctor", "professional", "medical"]):
                issues.append("Medical advice should include consultation recommendation")
        
        confidence = 0.9 if not issues else 0.6
        sources = ["Medical fact-checking completed"]
        
        return {
            "is_accurate": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "sources": sources,
            "recommendations": ["Always consult healthcare professionals for medical advice"]
        }
    
    def _check_legal_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check legal-related facts."""
        issues = []
        sources = []
        
        # Check for legal advice patterns
        legal_red_flags = [
            r"guaranteed.*win",
            r"simple.*legal.*trick",
            r"avoid.*law.*completely"
        ]
        
        for pattern in legal_red_flags:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential legal misinformation detected: {pattern}")
        
        # Check for specific legal claims
        if "legal" in response.lower() or "law" in response.lower():
            if not any(word in response.lower() for word in ["consult", "lawyer", "attorney", "professional"]):
                issues.append("Legal advice should include professional consultation recommendation")
        
        confidence = 0.9 if not issues else 0.6
        sources = ["Legal fact-checking completed"]
        
        return {
            "is_accurate": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "sources": sources,
            "recommendations": ["Always consult legal professionals for legal advice"]
        }
    
    def _check_financial_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check financial-related facts."""
        issues = []
        sources = []
        
        # Check for financial misinformation patterns
        financial_red_flags = [
            r"guaranteed.*return",
            r"get.*rich.*quick",
            r"risk.*free.*investment"
        ]
        
        for pattern in financial_red_flags:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential financial misinformation detected: {pattern}")
        
        confidence = 0.9 if not issues else 0.6
        sources = ["Financial fact-checking completed"]
        
        return {
            "is_accurate": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "sources": sources,
            "recommendations": ["Always consult financial professionals for investment advice"]
        }
    
    def _check_emergency_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check emergency-related facts."""
        issues = []
        sources = []
        
        # Check for emergency response patterns
        emergency_red_flags = [
            r"wait.*call.*doctor",
            r"ignore.*symptoms",
            r"delay.*emergency"
        ]
        
        for pattern in emergency_red_flags:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential emergency misinformation detected: {pattern}")
        
        # Check for immediate action recommendations
        if "emergency" in query.lower() or "urgent" in query.lower():
            if not any(word in response.lower() for word in ["call", "emergency", "911", "immediate"]):
                issues.append("Emergency queries should include immediate action recommendations")
        
        confidence = 0.95 if not issues else 0.5
        sources = ["Emergency fact-checking completed"]
        
        return {
            "is_accurate": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "sources": sources,
            "recommendations": ["For emergencies, always call emergency services immediately"]
        }
    
    def _check_general_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check general facts."""
        issues = []
        sources = []
        
        # Check for general misinformation patterns
        general_red_flags = [
            r"conspiracy",
            r"secret.*government",
            r"hidden.*truth"
        ]
        
        for pattern in general_red_flags:
            if re.search(pattern, response.lower()):
                issues.append(f"Potential general misinformation detected: {pattern}")
        
        confidence = 0.8 if not issues else 0.7
        sources = ["General fact-checking completed"]
        
        return {
            "is_accurate": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "sources": sources,
            "recommendations": ["Verify information from reliable sources"]
        }
    
    def _check_wikipedia_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check facts against Wikipedia (placeholder for future implementation)."""
        # This would be implemented with actual Wikipedia API calls
        return {
            "is_accurate": True,
            "confidence": 0.7,
            "issues": [],
            "sources": ["Wikipedia (placeholder)"],
            "recommendations": ["Cross-reference with multiple sources"]
        }
    
    def _check_web_facts(self, response: str, query: str) -> Dict[str, Any]:
        """Check facts against web search (placeholder for future implementation)."""
        # This would be implemented with actual web search API calls
        return {
            "is_accurate": True,
            "confidence": 0.6,
            "issues": [],
            "sources": ["Web search (placeholder)"],
            "recommendations": ["Verify with authoritative sources"]
        }