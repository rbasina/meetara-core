"""
Domain Categorizer for Meetara Core.
Centralized, efficient domain categorization using existing domain_config.yaml.
✅ NO DUPLICATION: Uses config_loader to get categories from domain_config.yaml
"""
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from app.core.logger import agent_logger
from app.core.config_loader import config_loader  # ✅ Use existing config


class DomainCategorizer:
    """Efficient domain categorization using existing domain_config.yaml."""
    
    # Domain authority mappings (category-specific, not domain-specific)
    DOMAIN_AUTHORITIES = {
        "healthcare": "NIMH, SAMHSA, WHO, CDC, or other health authorities",
        "legal_financial": "legal precedents, bar associations, CFPB, SEC, FINRA, or financial regulatory bodies",
        "emergency_crisis": "emergency services, crisis hotlines, or safety authorities",
        "business": "industry best practices, business standards, or professional associations",
        "education": "educational standards, academic research, or educational institutions",
        "technology": "industry standards, technical documentation, or professional organizations",
        "research": "academic journals, research institutions, or scholarly publications",
        "aerospace": "aviation authorities, transportation agencies, or aerospace standards",
        "industrial": "industry standards, manufacturing associations, or agricultural authorities",
        "specialized": "engineering standards, professional organizations, or technical authorities",
        "entertainment": "entertainment industry standards, media organizations, or cultural authorities",
        "culinary": "culinary institutes, food safety authorities, or restaurant associations",
        "default": "domain-specific authorities and evidence-based practices"
    }
    
    # Domain section headers (organized by category from domain_config.yaml)
    DOMAIN_SECTIONS = {
        "healthcare": [
            "**Emotional & Psychological Well-being**",
            "**Physical Health & Lifestyle**",
            "**Social & Environmental Support**",
            "**Practical Strategies & Techniques**",
            "**When to Seek Professional Help**"
        ],
        "legal_financial": [
            "**Legal/Financial Framework & Principles**",
            "**Practical Applications & Strategies**",
            "**Compliance & Regulatory Considerations**",
            "**Risk Management & Best Practices**",
            "**When to Consult a Professional**"
        ],
        "emergency_crisis": [
            "**Immediate Response & Actions**",
            "**Prevention & Preparedness**",
            "**Safety Protocols & Guidelines**",
            "**Recovery & Follow-up**",
            "**Emergency Contacts & Resources**"
        ],
        "business": [
            "**Core Concepts & Strategies**",
            "**Implementation & Best Practices**",
            "**Tools & Techniques**",
            "**Measuring Success & Metrics**",
            "**Common Challenges & Solutions**"
        ],
        "education": [
            "**Core Concepts & Fundamentals**",
            "**Effective Learning Strategies**",
            "**Practical Application & Practice**",
            "**Assessment & Progress Tracking**",
            "**Additional Resources & Support**"
        ],
        "technology": [
            "**Core Concepts & Fundamentals**",
            "**Implementation & Best Practices**",
            "**Tools & Technologies**",
            "**Common Patterns & Solutions**",
            "**Next Steps & Resources**"
        ],
        "creative": [
            "**Creative Foundations & Principles**",
            "**Techniques & Methodologies**",
            "**Tools & Resources**",
            "**Inspiring Examples & References**",
            "**Developing Your Craft**"
        ],
        "wellness": [
            "**Understanding the Fundamentals**",
            "**Practical Techniques & Approaches**",
            "**Integration into Daily Life**",
            "**Measuring Progress & Growth**",
            "**Seeking Additional Support**"
        ],
        "daily_life": [
            "**Key Principles & Approaches**",
            "**Practical Strategies & Tips**",
            "**Communication & Relationships**",
            "**Overcoming Challenges**",
            "**Long-term Success & Growth**"
        ],
        "travel": [
            "**Planning & Preparation**",
            "**Destination Highlights**",
            "**Practical Tips & Recommendations**",
            "**Cultural Considerations**",
            "**Resources & Additional Information**"
        ],
        "sports": [
            "**Fundamentals & Training Principles**",
            "**Techniques & Best Practices**",
            "**Equipment & Resources**",
            "**Performance & Progress Tracking**",
            "**Safety & Injury Prevention**"
        ],
        "research": [
            "**Research Methodology & Framework**",
            "**Data Collection & Analysis**",
            "**Academic Writing & Documentation**",
            "**Quality Assurance & Validation**",
            "**Academic Resources & References**"
        ],
        "industrial": [
            "**Production Processes & Methods**",
            "**Quality Control & Standards**",
            "**Equipment & Technology**",
            "**Safety & Environmental Considerations**",
            "**Industry Best Practices**"
        ],
        "specialized": [
            "**Core Principles & Theory**",
            "**Practical Applications & Implementation**",
            "**Advanced Techniques & Methodologies**",
            "**Quality & Validation Standards**",
            "**Expert Resources & References**"
        ],
        "entertainment": [
            "**Entertainment Formats & Genres**",
            "**Creative Development & Production**",
            "**Industry Insights & Trends**",
            "**Audience Engagement & Marketing**",
            "**Resources & Platforms**"
        ],
        "culinary": [
            "**Culinary Fundamentals & Techniques**",
            "**Recipe Development & Execution**",
            "**Kitchen Safety & Best Practices**",
            "**Food Quality & Presentation**",
            "**Culinary Resources & Inspiration**"
        ],
        "general_knowledge": [
            "**Essential Facts & Definitions**",
            "**Key Details & Context**",
            "**Historical or Global Perspective**",
            "**Practical Examples & Applications**",
            "**Further Resources & Next Steps**"
        ],
        "default": [
            "**Core Concepts & Fundamentals**",
            "**Practical Applications & Strategies**",
            "**Best Practices & Tips**",
            "**Common Considerations**",
            "**Additional Resources & Next Steps**"
        ]
    }
    
    def __init__(self):
        """Initialize domain categorizer using existing config_loader."""
        agent_logger.info("DomainCategorizer initialized - using domain_config.yaml for categories")
    
    @lru_cache(maxsize=256)
    def get_domain_category(self, domain: str) -> str:
        """
        Get category for a domain from domain_config.yaml (cached for performance).
        
        ✅ NO KEYWORD MATCHING: Directly uses config_loader.get_domain_config() 
        which returns the category from domain_config.yaml.
        
        Args:
            domain: Domain name (e.g., "academic_tutoring", "general_health")
            
        Returns:
            Category name from domain_config.yaml (e.g., "education", "healthcare")
        """
        if domain == "general_knowledge":
            return "general_knowledge"
        
        # ✅ Use existing config_loader to get category from domain_config.yaml
        domain_config = config_loader.get_domain_config(domain)
        category = domain_config.get('category', 'default')
        
        if category == 'default' and domain_config:
            # If category not found, try to infer from domain name as fallback
            domain_lower = domain.lower()
            if any(kw in domain_lower for kw in ['health', 'medical', 'nutrition', 'sleep', 'stress', 'mental']):
                return 'healthcare'
            elif any(kw in domain_lower for kw in ['legal', 'financial', 'insurance', 'real_estate']):
                return 'legal_financial'
            elif any(kw in domain_lower for kw in ['education', 'academic', 'teaching', 'learning', 'study']):
                return 'education'
            elif any(kw in domain_lower for kw in ['programming', 'software', 'tech', 'cyber', 'ai', 'data']):
                return 'technology'
            elif any(kw in domain_lower for kw in ['business', 'marketing', 'sales', 'entrepreneurship']):
                return 'business'
            elif any(kw in domain_lower for kw in ['emergency', 'crisis', 'disaster', 'safety']):
                return 'emergency_crisis'
        
        return category if category else 'default'
    
    def get_domain_sections(self, domain: str) -> List[str]:
        """
        Get section headers for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of section header strings
        """
        category = self.get_domain_category(domain)
        return self.DOMAIN_SECTIONS.get(category, self.DOMAIN_SECTIONS["default"])
    
    def get_domain_authorities(self, domain: str) -> str:
        """
        Get authority references for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Authority reference string
        """
        category = self.get_domain_category(domain)
        
        # Special handling for aerospace/transportation
        if any(kw in domain.lower() for kw in ["aerospace", "aeronautics", "automobile", "space", "transportation"]):
            return self.DOMAIN_AUTHORITIES.get("aerospace", self.DOMAIN_AUTHORITIES["default"])
        
        return self.DOMAIN_AUTHORITIES.get(category, self.DOMAIN_AUTHORITIES["default"])
    
    def is_medical_domain(self, domain: str) -> bool:
        """
        Check if domain is medical/healthcare related.
        
        Args:
            domain: Domain name
            
        Returns:
            True if medical domain
        """
        return self.get_domain_category(domain) == "healthcare"
    
    def clear_cache(self):
        """Clear LRU cache (useful for testing or config reloads)."""
        self.get_domain_category.cache_clear()


# Global instance
_domain_categorizer: Optional[DomainCategorizer] = None


def get_domain_categorizer() -> DomainCategorizer:
    """Get or create global domain categorizer instance."""
    global _domain_categorizer
    if _domain_categorizer is None:
        _domain_categorizer = DomainCategorizer()
    return _domain_categorizer

