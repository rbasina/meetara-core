"""
Test Model Mock Script for Meetara Core.

This script provides mock model responses for testing the configuration-driven
domain system without requiring actual model loading.
"""
import json
import time
from typing import Dict, Any, List
from app.core.logger import agent_logger
from app.core.config_loader import config_loader


class ModelMockTester:
    """Mock model tester for development and testing."""
    
    def __init__(self):
        """Initialize the mock tester."""
        self.mock_responses = self._load_mock_responses()
        agent_logger.info("Model mock tester initialized")
    
    def _load_mock_responses(self) -> Dict[str, Any]:
        """Load mock responses for different domains."""
        return {
            "healthcare": {
                "general_health": "I understand you're asking about health. For general health concerns, it's important to consult with a healthcare professional. I can provide general information, but specific medical advice should come from qualified doctors.",
                "mental_health": "Mental health is crucial for overall wellbeing. If you're experiencing mental health challenges, consider speaking with a mental health professional. There are many resources available for support.",
                "emergency_care": "If this is a medical emergency, please call emergency services immediately. For urgent health concerns, contact your healthcare provider or visit the nearest emergency room."
            },
            "business": {
                "entrepreneurship": "Starting a business requires careful planning and research. Consider your market, funding options, and legal requirements. Many successful entrepreneurs start with a solid business plan.",
                "marketing": "Effective marketing involves understanding your target audience and creating compelling messaging. Digital marketing, social media, and traditional advertising all have their place in a comprehensive strategy.",
                "project_management": "Good project management involves clear goals, timelines, and communication. Tools like project management software can help track progress and keep teams aligned."
            },
            "education": {
                "academic_tutoring": "Academic success often comes from good study habits and seeking help when needed. Consider working with tutors, forming study groups, or using online resources to strengthen your understanding.",
                "study_techniques": "Effective study techniques include active learning, spaced repetition, and regular review. Find methods that work best for your learning style and the subject matter.",
                "career_guidance": "Career development involves continuous learning and networking. Consider your interests, skills, and market opportunities when planning your career path."
            },
            "technology": {
                "programming": "Programming requires practice and patience. Start with fundamentals, work on small projects, and gradually build your skills. There are many excellent online resources for learning to code.",
                "ai_ml": "Artificial Intelligence and Machine Learning are rapidly evolving fields. Understanding the basics of algorithms, data structures, and mathematical concepts is essential for working in AI/ML.",
                "cybersecurity": "Cybersecurity is critical in today's digital world. Understanding basic security principles, staying updated on threats, and implementing good security practices is important for everyone."
            },
            "daily_life": {
                "parenting": "Parenting is a rewarding but challenging journey. Every child is different, and what works for one may not work for another. Trust your instincts and don't hesitate to seek advice from other parents or professionals.",
                "relationships": "Healthy relationships are built on communication, trust, and mutual respect. Open dialogue and understanding each other's needs are key to maintaining strong connections.",
                "time_management": "Effective time management involves prioritizing tasks, setting realistic goals, and learning to say no when necessary. Tools like calendars and to-do lists can help stay organized."
            },
            "creative": {
                "writing": "Writing is a skill that improves with practice. Read widely, write regularly, and don't be afraid to share your work for feedback. Every writer has their own unique voice and style.",
                "art_appreciation": "Art appreciation involves understanding context, technique, and personal interpretation. Visit galleries, read about artists, and develop your own aesthetic preferences.",
                "music": "Music appreciation can be deepened through active listening, learning about different genres, and understanding the cultural and historical context of various musical traditions."
            },
            "legal_financial": {
                "legal_assistance": "Legal matters can be complex and vary by jurisdiction. For specific legal advice, consult with a qualified attorney who can provide guidance based on your particular situation and local laws.",
                "financial_planning": "Financial planning involves budgeting, saving, investing, and planning for the future. Consider working with a financial advisor to develop a comprehensive plan tailored to your goals.",
                "insurance": "Insurance provides protection against various risks. Understanding your needs and comparing different policies can help you make informed decisions about coverage."
            },
            "emergency_crisis": {
                "crisis_management": "In crisis situations, stay calm and follow established emergency procedures. Contact appropriate authorities and follow their guidance. Preparation and clear communication are key to effective crisis management.",
                "safety_security": "Personal safety involves awareness, preparation, and common sense. Learn basic safety procedures and stay informed about potential risks in your environment."
            }
        }
    
    def get_mock_response(self, domain: str, query: str) -> Dict[str, Any]:
        """Get a mock response for a given domain and query."""
        try:
            # Get domain configuration
            domain_config = config_loader.get_domain_config(domain)
            tier = domain_config.get('category_tier', 'quality')
            
            # Get mock response based on domain category
            category = self._get_domain_category(domain)
            mock_response = self.mock_responses.get(category, {}).get(domain, 
                "I understand your question. This is a mock response for testing purposes. In a real implementation, this would be a domain-specific response.")
            
            # Simulate processing time based on tier
            processing_time = self._get_processing_time(tier)
            time.sleep(processing_time)
            
            # Generate confidence score based on domain match
            confidence = self._calculate_confidence(domain, query, tier)
            
            return {
                "response": mock_response,
                "domain": domain,
                "tier": tier,
                "confidence": confidence,
                "processing_time": processing_time,
                "is_mock": True,
                "model_used": domain_config.get('model', 'microsoft/microsoft--DialoGPT-small')
            }
            
        except Exception as e:
            agent_logger.error(f"Error in mock response generation: {e}")
            return {
                "response": "I apologize, but I'm unable to provide a response at the moment. This is a mock testing environment.",
                "domain": domain,
                "tier": "quality",
                "confidence": 0.5,
                "processing_time": 0.1,
                "is_mock": True,
                "error": str(e)
            }
    
    def _get_domain_category(self, domain: str) -> str:
        """Get the category for a domain."""
        domain_categories = {
            "healthcare": ["general_health", "mental_health", "emergency_care", "nutrition", "sleep", "stress_management"],
            "business": ["entrepreneurship", "marketing", "project_management", "team_leadership", "financial_planning"],
            "education": ["academic_tutoring", "study_techniques", "career_guidance", "exam_preparation"],
            "technology": ["programming", "ai_ml", "cybersecurity", "software_development"],
            "daily_life": ["parenting", "relationships", "time_management", "communication"],
            "creative": ["writing", "art_appreciation", "music", "storytelling"],
            "legal_financial": ["legal_assistance", "financial_planning", "insurance", "real_estate"],
            "emergency_crisis": ["crisis_management", "safety_security", "emergency_response"]
        }
        
        for category, domains in domain_categories.items():
            if domain in domains:
                return category
        
        return "general"
    
    def _get_processing_time(self, tier: str) -> float:
        """Get processing time based on tier."""
        tier_times = {
            "safety_critical": 0.1,  # Fastest for critical domains
            "expert": 0.2,           # Medium for expert domains
            "quality": 0.3           # Standard for quality domains
        }
        return tier_times.get(tier, 0.3)
    
    def _calculate_confidence(self, domain: str, query: str, tier: str) -> float:
        """Calculate confidence score based on domain and query."""
        base_confidence = 0.8
        
        # Adjust based on tier
        tier_adjustments = {
            "safety_critical": 0.1,  # Higher confidence for critical domains
            "expert": 0.0,           # Standard confidence for expert domains
            "quality": -0.1           # Slightly lower for quality domains
        }
        
        confidence = base_confidence + tier_adjustments.get(tier, 0.0)
        
        # Adjust based on query length and complexity
        query_words = len(query.split())
        if query_words > 10:
            confidence += 0.05  # More detailed queries get higher confidence
        elif query_words < 3:
            confidence -= 0.1   # Very short queries get lower confidence
        
        return max(0.1, min(1.0, confidence))
    
    def test_domain_selection(self, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Test domain selection with various queries."""
        results = []
        
        for test_case in test_queries:
            query = test_case["query"]
            expected_domain = test_case.get("expected_domain", "general")
            
            # Simulate domain selection
            urgency_info = config_loader.detect_urgency(query)
            
            # Mock domain selection logic
            selected_domain = self._mock_domain_selection(query, urgency_info)
            
            result = {
                "query": query,
                "selected_domain": selected_domain,
                "expected_domain": expected_domain,
                "urgency_info": urgency_info,
                "is_correct": selected_domain == expected_domain
            }
            
            results.append(result)
        
        # Calculate accuracy
        correct_selections = sum(1 for r in results if r["is_correct"])
        accuracy = correct_selections / len(results) if results else 0.0
        
        return {
            "results": results,
            "accuracy": accuracy,
            "total_tests": len(results),
            "correct_selections": correct_selections
        }
    
    def _mock_domain_selection(self, query: str, urgency_info: Dict[str, Any]) -> str:
        """Mock domain selection logic."""
        query_lower = query.lower()
        
        # Simple keyword-based domain selection
        domain_keywords = {
            "general_health": ["health", "medical", "doctor", "symptoms"],
            "mental_health": ["mental", "therapy", "anxiety", "depression"],
            "emergency_care": ["emergency", "urgent", "crisis", "help"],
            "entrepreneurship": ["business", "startup", "entrepreneur"],
            "programming": ["code", "programming", "software", "development"],
            "parenting": ["child", "parent", "family", "kids"],
            "relationships": ["relationship", "dating", "marriage", "love"]
        }
        
        # Find best matching domain
        best_domain = "general"
        best_score = 0.0
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        
        return best_domain
    
    def test_validation_system(self, test_responses: List[Dict[str, str]]) -> Dict[str, Any]:
        """Test the validation system with various responses."""
        results = []
        
        for test_case in test_responses:
            response = test_case["response"]
            domain = test_case["domain"]
            query = test_case["query"]
            
            # Mock validation checks
            validation_result = self._mock_validation(response, domain, query)
            
            result = {
                "response": response[:100] + "..." if len(response) > 100 else response,
                "domain": domain,
                "is_valid": validation_result["is_valid"],
                "confidence": validation_result["confidence"],
                "issues": validation_result["issues"]
            }
            
            results.append(result)
        
        return {
            "validation_results": results,
            "total_tested": len(results),
            "valid_responses": sum(1 for r in results if r["is_valid"])
        }
    
    def _mock_validation(self, response: str, domain: str, query: str) -> Dict[str, Any]:
        """Mock validation checks."""
        issues = []
        confidence = 0.8
        
        # Check for banned content
        banned_words = ["hack", "illegal", "harm", "dangerous"]
        for word in banned_words:
            if word in response.lower():
                issues.append(f"Banned content detected: {word}")
                confidence -= 0.3
        
        # Check for vague responses
        vague_words = ["it depends", "it varies", "generally"]
        vague_count = sum(1 for word in vague_words if word in response.lower())
        if vague_count > 2:
            issues.append("Too many vague statements")
            confidence -= 0.2
        
        # Domain-specific checks
        if domain in ["general_health", "mental_health"]:
            if "consult" not in response.lower() and "doctor" not in response.lower():
                if any(word in response.lower() for word in ["treatment", "medicine", "diagnosis"]):
                    issues.append("Medical response should include consultation recommendation")
                    confidence -= 0.1
        
        return {
            "is_valid": len(issues) == 0,
            "confidence": max(0.1, confidence),
            "issues": issues
        }


def run_mock_tests():
    """Run comprehensive mock tests."""
    tester = ModelMockTester()
    
    # Test queries
    test_queries = [
        {"query": "I have chest pain", "expected_domain": "emergency_care"},
        {"query": "How to start a business", "expected_domain": "entrepreneurship"},
        {"query": "Learn programming", "expected_domain": "programming"},
        {"query": "Parenting advice", "expected_domain": "parenting"},
        {"query": "Mental health support", "expected_domain": "mental_health"}
    ]
    
    # Test responses
    test_responses = [
        {"response": "You should consult a doctor immediately for chest pain.", "domain": "emergency_care", "query": "I have chest pain"},
        {"response": "Starting a business requires planning and research.", "domain": "entrepreneurship", "query": "How to start a business"},
        {"response": "Programming can be learned through practice and study.", "domain": "programming", "query": "Learn programming"}
    ]
    
    # Run tests
    domain_results = tester.test_domain_selection(test_queries)
    validation_results = tester.test_validation_system(test_responses)
    
    # Print results
    print("=== Mock Test Results ===")
    print(f"Domain Selection Accuracy: {domain_results['accuracy']:.2%}")
    print(f"Validation Results: {validation_results['valid_responses']}/{validation_results['total_tested']} valid")
    
    return {
        "domain_selection": domain_results,
        "validation": validation_results
    }


if __name__ == "__main__":
    run_mock_tests() 