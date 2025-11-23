"""
Domain Mapper for Meetara Core.

⚠️ LEGACY MODULE: This module provides domain descriptions and search functionality.
For domain categorization (used by LLM processors), use `domain_categorizer.py` instead.

This module is kept for:
- Domain descriptions (DOMAIN_DESCRIPTIONS)
- Domain search functionality
- Test scripts compatibility

For production use, prefer `domain_categorizer.py` which uses domain_config.yaml.
"""
from typing import Dict, List, Any, Optional
from enum import Enum


class DomainCategory(Enum):
    """Domain categories for organization."""
    ACADEMIC = "academic"
    TECHNOLOGY = "technology"
    HEALTH_WELLNESS = "health_wellness"
    BUSINESS_PROFESSIONAL = "business_professional"
    FINANCIAL = "financial"
    LEGAL = "legal"
    CREATIVE_ARTS = "creative_arts"
    PERSONAL_DEVELOPMENT = "personal_development"
    RELATIONSHIPS_SOCIAL = "relationships_social"
    TRAVEL_TRANSPORTATION = "travel_transportation"
    HOME_LIFESTYLE = "home_lifestyle"
    CRISIS_EMERGENCY = "crisis_emergency"
    TECHNOLOGY_SUPPORT = "technology_support"
    RESEARCH_SCIENCE = "research_science"
    INDUSTRY_SPECIFIC = "industry_specific"
    SPECIALIZED_AREAS = "specialized_areas"
    RECREATION_ENTERTAINMENT = "recreation_entertainment"
    WRITING_COMMUNICATION = "writing_communication"
    PLANNING_ORGANIZATION = "planning_organization"


class DomainMapper:
    """Utility for mapping and categorizing domains."""
    
    # Domain categorization
    DOMAIN_CATEGORIES = {
        DomainCategory.ACADEMIC: [
            "academic_tutoring",
            "academic_tutoring_research", 
            "education",
            "educational_technology",
            "exam_preparation",
            "study_techniques",
            "teaching"
        ],
        
        DomainCategory.TECHNOLOGY: [
            "artificial_intelligence",
            "data_science", 
            "machine_learning",
            "programming",
            "software_development",
            "cybersecurity",
            "engineering",
            "aerospace_engineering",
            "aeronautics",
            "space_technology"
        ],
        
        DomainCategory.HEALTH_WELLNESS: [
            "general_health",
            "mental_health",
            "fitness_healthcare",
            "chronic_conditions",
            "emergency_care",
            "emergency_response",
            "preventive_care",
            "senior_health",
            "women_health",
            "medication_management",
            "nutrition",
            "sleep",
            "stress_management",
            "yoga"
        ],
        
        DomainCategory.BUSINESS_PROFESSIONAL: [
            "business",
            "career_guidance",
            "consulting",
            "customer_service",
            "hr_management",
            "marketing",
            "sales",
            "project_management",
            "operations",
            "team_leadership",
            "remote_work",
            "work_life_balance"
        ],
        
        DomainCategory.FINANCIAL: [
            "financial",
            "financial_planning",
            "insurance",
            "real_estate"
        ],
        
        DomainCategory.LEGAL: [
            "legal",
            "legal_assistance",
            "legal_business"
        ],
        
        DomainCategory.CREATIVE_ARTS: [
            "art_appreciation",
            "creative_writing",
            "content_creation",
            "music",
            "photography",
            "storytelling"
        ],
        
        DomainCategory.PERSONAL_DEVELOPMENT: [
            "life_coaching",
            "personal_assistant",
            "skill_development",
            "time_management",
            "communication",
            "conflict_resolution",
            "decision_making"
        ],
        
        DomainCategory.RELATIONSHIPS_SOCIAL: [
            "relationships",
            "parenting",
            "social_support",
            "social_media",
            "social_media_management"
        ],
        
        DomainCategory.TRAVEL_TRANSPORTATION: [
            "travel_tourism",
            "transportation"
        ],
        
        DomainCategory.HOME_LIFESTYLE: [
            "home_management",
            "shopping"
        ],
        
        DomainCategory.CRISIS_EMERGENCY: [
            "crisis_management",
            "disaster_preparedness",
            "safety_security"
        ],
        
        DomainCategory.TECHNOLOGY_SUPPORT: [
            "tech_support",
            "digital_literacy"
        ],
        
        DomainCategory.RESEARCH_SCIENCE: [
            "research",
            "research_assistance",
            "scientific_research"
        ],
        
        DomainCategory.INDUSTRY_SPECIFIC: [
            "agriculture",
            "automobile",
            "manufacturing",
            "data_analysis"
        ],
        
        DomainCategory.SPECIALIZED_AREAS: [
            "design_thinking",
            "entrepreneurship",
            "mythology",
            "spiritual",
            "history",
            "politics"
        ],
        
        DomainCategory.RECREATION_ENTERTAINMENT: [
            "sports_recreation",
            "animals"
        ],
        
        DomainCategory.WRITING_COMMUNICATION: [
            "writing"
        ],
        
        DomainCategory.PLANNING_ORGANIZATION: [
            "planning"
        ]
    }
    
    # Domain descriptions for better understanding
    DOMAIN_DESCRIPTIONS = {
        # Academic & Education
        "academic_tutoring": "Academic tutoring and homework help for students",
        "academic_tutoring_research": "Research assistance and academic writing support",
        "education": "General education and learning resources",
        "educational_technology": "Digital learning and educational technology",
        "exam_preparation": "Test preparation and study strategies",
        "study_techniques": "Effective study methods and learning strategies",
        "teaching": "Teaching methodologies and classroom management",
        
        # Technology & Engineering
        "artificial_intelligence": "AI, machine learning, and neural networks",
        "data_science": "Data analysis, statistics, and analytics",
        "machine_learning": "ML algorithms, models, and predictions",
        "programming": "Software development and coding",
        "software_development": "Application development and software engineering",
        "cybersecurity": "Digital security and protection",
        "engineering": "General engineering and technical design",
        "aerospace_engineering": "Aviation and aerospace technology",
        "aeronautics": "Flight and aircraft technology",
        "space_technology": "Space exploration and satellite technology",
        
        # Health & Wellness
        "general_health": "General health and medical information",
        "mental_health": "Psychology, therapy, and mental wellness",
        "fitness_healthcare": "Physical fitness and exercise",
        "chronic_conditions": "Long-term health conditions and management",
        "emergency_care": "Emergency medical care and first aid",
        "emergency_response": "Crisis response and emergency management",
        "preventive_care": "Health prevention and wellness screening",
        "senior_health": "Elderly care and aging health",
        "women_health": "Women's health and reproductive care",
        "medication_management": "Pharmaceutical care and medication",
        "nutrition": "Diet and nutritional guidance",
        "sleep": "Sleep health and rest management",
        "stress_management": "Stress relief and relaxation techniques",
        "yoga": "Yoga, meditation, and mindfulness",
        
        # Business & Professional
        "business": "General business and entrepreneurship",
        "career_guidance": "Career planning and job search",
        "consulting": "Professional consulting and advisory",
        "customer_service": "Customer support and service",
        "hr_management": "Human resources and personnel management",
        "marketing": "Marketing and advertising strategies",
        "sales": "Sales techniques and revenue generation",
        "project_management": "Project planning and execution",
        "operations": "Business operations and process management",
        "team_leadership": "Leadership and team management",
        "remote_work": "Remote work and telecommuting",
        "work_life_balance": "Work-life balance and time management",
        
        # Financial
        "financial": "General finance and money management",
        "financial_planning": "Personal financial planning and budgeting",
        "insurance": "Insurance coverage and risk management",
        "real_estate": "Property and real estate transactions",
        
        # Legal
        "legal": "General legal advice and law",
        "legal_assistance": "Legal help and attorney services",
        "legal_business": "Corporate law and business legal matters",
        
        # Creative & Arts
        "art_appreciation": "Art appreciation and cultural understanding",
        "creative_writing": "Creative writing and storytelling",
        "content_creation": "Digital content creation and media",
        "music": "Music appreciation and musical education",
        "photography": "Photography and visual arts",
        "storytelling": "Narrative and storytelling techniques",
        
        # Personal Development
        "life_coaching": "Life coaching and personal development",
        "personal_assistant": "Personal assistance and support",
        "skill_development": "Skill building and personal growth",
        "time_management": "Productivity and time organization",
        "communication": "Communication skills and interpersonal",
        "conflict_resolution": "Conflict management and mediation",
        "decision_making": "Decision analysis and problem solving",
        
        # Relationships & Social
        "relationships": "Interpersonal relationships and dating",
        "parenting": "Child rearing and family management",
        "social_support": "Community support and social networks",
        "social_media": "Social media usage and online presence",
        "social_media_management": "Social media marketing and management",
        
        # Travel & Transportation
        "travel_tourism": "Travel planning and tourism",
        "transportation": "Transportation and mobility solutions",
        
        # Home & Lifestyle
        "home_management": "Household management and domestic care",
        "shopping": "Consumer shopping and retail guidance",
        
        # Crisis & Emergency
        "crisis_management": "Crisis handling and emergency response",
        "disaster_preparedness": "Disaster preparation and safety",
        "safety_security": "Personal safety and security measures",
        
        # Technology Support
        "tech_support": "Technical support and computer help",
        "digital_literacy": "Digital skills and technology education",
        
        # Research & Science
        "research": "Research methodologies and academic inquiry",
        "research_assistance": "Research support and academic help",
        "scientific_research": "Scientific inquiry and laboratory work",
        
        # Industry Specific
        "agriculture": "Farming and agricultural practices",
        "automobile": "Automotive care and vehicle maintenance",
        "manufacturing": "Manufacturing and production processes",
        "data_analysis": "Data analytics and business intelligence",
        
        # Specialized Areas
        "design_thinking": "Innovation and creative problem solving",
        "entrepreneurship": "Startup business and innovation",
        "mythology": "Cultural myths, legends, and folklore",
        "spiritual": "Spirituality and religious practices",
        "history": "Historical knowledge and cultural heritage",
        "politics": "Political understanding and civic engagement",
        
        # Recreation & Entertainment
        "sports_recreation": "Sports, games, and recreational activities",
        "animals": "Pet care and animal welfare",
        
        # Writing & Communication
        "writing": "Writing skills and literary composition",
        
        # Planning & Organization
        "planning": "Strategic planning and organization"
    }
    
    @classmethod
    def get_domain_category(cls, domain: str) -> Optional[DomainCategory]:
        """Get the category for a specific domain."""
        for category, domains in cls.DOMAIN_CATEGORIES.items():
            if domain in domains:
                return category
        return None
    
    @classmethod
    def get_domains_by_category(cls, category: DomainCategory) -> List[str]:
        """Get all domains in a specific category."""
        return cls.DOMAIN_CATEGORIES.get(category, [])
    
    @classmethod
    def get_all_domains(cls) -> List[str]:
        """Get all available domains."""
        all_domains = []
        for domains in cls.DOMAIN_CATEGORIES.values():
            all_domains.extend(domains)
        return sorted(all_domains)
    
    @classmethod
    def get_domain_description(cls, domain: str) -> str:
        """Get description for a specific domain."""
        return cls.DOMAIN_DESCRIPTIONS.get(domain, "Domain assistance and support")
    
    @classmethod
    def get_category_summary(cls) -> Dict[str, Any]:
        """Get summary of all categories and domains."""
        summary = {}
        for category in DomainCategory:
            domains = cls.get_domains_by_category(category)
            summary[category.value] = {
                "count": len(domains),
                "domains": domains,
                "description": category.value.replace("_", " ").title()
            }
        return summary
    
    @classmethod
    def search_domains(cls, query: str) -> List[str]:
        """Search domains by keyword."""
        query_lower = query.lower()
        matching_domains = []
        
        for domain, description in cls.DOMAIN_DESCRIPTIONS.items():
            if (query_lower in domain.lower() or 
                query_lower in description.lower()):
                matching_domains.append(domain)
        
        return matching_domains
    
    @classmethod
    def get_related_domains(cls, domain: str) -> List[str]:
        """Get related domains in the same category."""
        category = cls.get_domain_category(domain)
        if category:
            domains = cls.get_domains_by_category(category)
            return [d for d in domains if d != domain]
        return [] 