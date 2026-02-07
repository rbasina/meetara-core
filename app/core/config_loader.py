"""
Configuration Loader for Meetara Core.

This module handles loading and managing domain configurations, tier settings,
and validation requirements from YAML files.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from app.utils.yaml_loader import load_yaml_file  # ✅ Use centralized YAML loader


class ConfigLoader:
    """Configuration loader for Meetara Core domain and tier settings."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.domain_config = None
        self.tier_config = None
        self.model_config = None
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files using centralized YAML loader."""
        try:
            # ✅ OPTIMIZED: Use centralized YAML loader utility
            domain_config_path = self.config_dir / "domain_config.yaml"
            self.domain_config = load_yaml_file(domain_config_path, default={})
            if self.domain_config:
                logger.info("Domain configuration loaded successfully")
            
            tier_config_path = self.config_dir / "tier_config.yaml"
            self.tier_config = load_yaml_file(tier_config_path, default={})
            if self.tier_config:
                logger.info("Tier configuration loaded successfully")
            
            keywords_path = self.config_dir / "domain_keywords.yaml"
            self.domain_keywords = load_yaml_file(keywords_path, default={})
            if self.domain_keywords:
                logger.info("Domain keywords loaded successfully")
            
            # ✅ Load model configuration from model_config.yaml
            model_config_path = self.config_dir / "model_config.yaml"
            self.model_config = load_yaml_file(model_config_path, default={})
            if self.model_config:
                logger.info(f"Model configuration loaded: {len(self.model_config.get('models', {}))} models available")
                
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            self.domain_config = {}
            self.tier_config = {}
            self.domain_keywords = {}
            self.model_config = {}
    
    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for a specific domain.
        
        Args:
            domain: Domain name to get configuration for
            
        Returns:
            Domain configuration dictionary
        """
        if not self.domain_config:
            return {}
        
        # Search through all categories for the domain
        for category, category_config in self.domain_config.get('categories', {}).items():
            domains = category_config.get('domains', {})
            if domain in domains:
                domain_config = domains[domain].copy()
                # Add category-level settings
                domain_config['category'] = category
                domain_config['category_tier'] = category_config.get('tier', 'quality')
                domain_config['category_validation_level'] = category_config.get('validation_level', 'standard')
                return domain_config
        
        # Return default configuration if domain not found
        return self.domain_config.get('default', {})
    
    def get_all_domains(self) -> List[str]:
        """Get list of all available domains.
        
        Returns:
            List of domain names
        """
        domains = []
        if not self.domain_config:
            return domains
        
        for category_config in self.domain_config.get('categories', {}).values():
            domains.extend(category_config.get('domains', {}).keys())
        
        return domains
    
    def get_all_category_names(self) -> List[str]:
        """Get list of all category names (e.g. business, education, healthcare).
        
        Useful for batch upload: --domain business can target vectorstore/business/
        even though 'business' is a category, not a leaf domain.
        
        Returns:
            List of category names
        """
        if not self.domain_config:
            return []
        return list(self.domain_config.get('categories', {}).keys())
    
    def get_tier_config(self, tier: str) -> Dict[str, Any]:
        """Get configuration for a specific tier.
        
        Args:
            tier: Tier name (safety_critical_domains, expert_domains, quality_domains)
            
        Returns:
            Tier configuration dictionary
        """
        if not self.tier_config:
            return {}
        
        return self.tier_config.get('tiers', {}).get(tier, {})
    
    def get_validation_requirements(self, tier: str) -> Dict[str, Any]:
        """Get validation requirements for a tier.
        
        Args:
            tier: Tier name
            
        Returns:
            Validation requirements dictionary
        """
        if not self.tier_config:
            return {}
        
        tier_name = tier.replace('_domains', '')
        return self.tier_config.get('validation_requirements', {}).get(tier_name, {})
    
    def get_matching_weights(self, query_type: str = 'default') -> Dict[str, float]:
        """Get hybrid matching weights for a query type.
        
        Args:
            query_type: Type of query (default, emergency, technical, creative)
            
        Returns:
            Dictionary with keyword, semantic, and context weights
        """
        if not self.tier_config:
            return {'keywords': 0.30, 'semantic': 0.50, 'context': 0.20}
        
        return self.tier_config.get('matching_weights', {}).get(query_type, {
            'keywords': 0.30,
            'semantic': 0.50,
            'context': 0.20
        })
    
    def get_urgency_patterns(self) -> Dict[str, Any]:
        """Get urgency detection patterns.
        
        Returns:
            Dictionary with urgency patterns
        """
        if not self.tier_config:
            return {}
        
        return self.tier_config.get('urgency_patterns', {})
    
    def detect_urgency(self, query: str) -> Dict[str, Any]:
        """Detect urgency in a query.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with urgency information
        """
        query_lower = query.lower()
        urgency_patterns = self.get_urgency_patterns()
        
        urgency_info = {
            'is_urgent': False,
            'urgency_type': None,
            'urgency_score': 0.0,
            'detected_patterns': []
        }
        
        # Check emergency keywords
        emergency_keywords = urgency_patterns.get('emergency_keywords', [])
        for keyword in emergency_keywords:
            if keyword in query_lower:
                urgency_info['is_urgent'] = True
                urgency_info['urgency_type'] = 'emergency'
                urgency_info['urgency_score'] += 0.5
                urgency_info['detected_patterns'].append(keyword)
        
        # Check time-based urgency
        time_keywords = urgency_patterns.get('time_based_urgency', [])
        for keyword in time_keywords:
            if keyword in query_lower:
                urgency_info['is_urgent'] = True
                urgency_info['urgency_type'] = 'time_based'
                urgency_info['urgency_score'] += 0.3
                urgency_info['detected_patterns'].append(keyword)
        
        # Check domain-specific cues
        domain_cues = urgency_patterns.get('domain_specific_cues', {})
        for domain, cues in domain_cues.items():
            for cue in cues:
                if cue in query_lower:
                    urgency_info['is_urgent'] = True
                    urgency_info['urgency_type'] = f'domain_specific_{domain}'
                    urgency_info['urgency_score'] += 0.4
                    urgency_info['detected_patterns'].append(cue)
        
        # Normalize urgency score
        urgency_info['urgency_score'] = min(urgency_info['urgency_score'], 1.0)
        
        return urgency_info
    
    def get_stop_words(self) -> set:
        """Get stop words for keyword extraction.
        
        Returns:
            Set of stop words (lowercase) to filter during keyword extraction
        """
        if hasattr(self, 'domain_keywords') and self.domain_keywords:
            stop_words_list = self.domain_keywords.get('stop_words', [])
            if stop_words_list:
                return set(word.lower() for word in stop_words_list)
        
        # Fallback to default stop words if config not found
        logger.warning("Stop words not found in config, using defaults")
        return {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'where', 'when', 'why', 'how', 'if', 'then', 'else',
            'so', 'than', 'more', 'most', 'very', 'just', 'only', 'also', 'too', 'not', 'no', 'yes',
            'and', 'or', 'but', 'want', 'many', 'each', 'around', 'show', 'me', 'image'
        }
    
    def get_fallback_message(self, domain: str) -> str:
        """Get fallback message for a domain when models fail.
        
        ✅ CONFIG-DRIVEN: Loads from domain_config.yaml instead of hardcoding.
        
        Args:
            domain: Domain name
            
        Returns:
            Fallback message string
        """
        domain_config = self.get_domain_config(domain)
        fallback_msg = domain_config.get('fallback_message')
        
        if fallback_msg:
            return fallback_msg
        
        # Try category-level fallback
        category = domain_config.get('category')
        if category and self.domain_config:
            category_config = self.domain_config.get('categories', {}).get(category, {})
            category_fallback = category_config.get('fallback_message')
            if category_fallback:
                return category_fallback
        
        # Default fallback if not configured
        default_fallback = self.domain_config.get('default', {}).get('fallback_message')
        if default_fallback:
            return default_fallback
        
        return "I'm currently initializing my specialized knowledge base. Please give me a moment to load the appropriate resources for your question."
    
    def get_domain_keywords(self, domain: str) -> List[str]:
        """Get comprehensive keywords for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of comprehensive keywords for the domain
        """
        # First try to get keywords from comprehensive keywords file
        if hasattr(self, 'domain_keywords') and self.domain_keywords:
            domain_keywords = self.domain_keywords.get('domains', {}).get(domain, {})
            if domain_keywords and 'keywords' in domain_keywords:
                return domain_keywords['keywords']
        
        # Fallback to domain config keywords
        domain_config = self.get_domain_config(domain)
        return domain_config.get('keywords', [])
    
    def get_domain_model(self, domain: str) -> str:
        """Get model assignment for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Model name for the domain
        """
        domain_config = self.get_domain_config(domain)
        return domain_config.get('model', 'microsoft/microsoft--DialoGPT-small')
    
    def get_academic_domains(self) -> List[str]:
        """Get all domains that belong to academic/educational categories.
        
        Returns:
            List of academic domain names
        """
        academic_domains = []
        if not self.domain_config:
            return academic_domains
        
        # Get domains from education category
        education_category = self.domain_config.get('categories', {}).get('education', {})
        if education_category:
            academic_domains.extend(education_category.get('domains', {}).keys())
        
        # Get domains from research_academic category
        research_category = self.domain_config.get('categories', {}).get('research_academic', {})
        if research_category:
            academic_domains.extend(research_category.get('domains', {}).keys())
        
        # Also check for science-related categories that might be academic
        # This includes any category that might contain academic domains
        science_keywords = ['science', 'mathematics', 'physics', 'chemistry', 'biology']
        for category_name, category_config in self.domain_config.get('categories', {}).items():
            category_lower = category_name.lower()
            # Check if category name contains academic keywords
            if any(keyword in category_lower for keyword in science_keywords):
                academic_domains.extend(category_config.get('domains', {}).keys())
        
        return list(set(academic_domains))  # Remove duplicates
    
    def requires_validation(self, domain: str) -> bool:
        """Check if a domain requires validation.
        
        Args:
            domain: Domain name
            
        Returns:
            True if domain requires validation
        """
        domain_config = self.get_domain_config(domain)
        return domain_config.get('requires_validation', False)
    
    def get_domain_tier(self, domain: str) -> str:
        """Get tier for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Tier name for the domain
        """
        domain_config = self.get_domain_config(domain)
        return domain_config.get('category_tier', 'quality')
    
    def reload_configurations(self):
        """Reload all configuration files."""
        logger.info("Reloading configurations...")
        self._load_configurations()
    
    # ============================================
    # MODEL CONFIGURATION METHODS
    # ============================================
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get all available models with their configurations.
        
        Returns:
            Dictionary of model configurations
        """
        if not self.model_config:
            return {}
        return self.model_config.get('models', {})
    
    def get_model_config(self, model_id: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_id: Model identifier (e.g., 'meetara-1.7b')
            
        Returns:
            Model configuration dictionary
        """
        models = self.get_available_models()
        return models.get(model_id, {})
    
    def get_default_model(self) -> str:
        """Get the default model ID.
        
        Returns:
            Default model identifier
        """
        models = self.get_available_models()
        for model_id, config in models.items():
            if config.get('default', False):
                return model_id
        return "meetara-1.7b"  # Fallback
    
    def get_model_for_domain(self, domain: str) -> str:
        """Get recommended model for a specific domain.
        
        Uses the tier mapping from model_config.yaml to select the best model.
        
        Args:
            domain: Domain name
            
        Returns:
            Model identifier
        """
        if not self.model_config:
            return self.get_default_model()
        
        # Get domain tier
        domain_tier = self.get_domain_tier(domain)
        
        # Get tier mapping from model config
        selection_strategy = self.model_config.get('selection_strategy', {})
        
        if not selection_strategy.get('auto_select', True):
            return selection_strategy.get('default_model', self.get_default_model())
        
        tier_mapping = selection_strategy.get('tier_mapping', {})
        
        # Map tier to model
        if domain_tier in tier_mapping:
            return tier_mapping[domain_tier]
        
        return selection_strategy.get('default_model', self.get_default_model())
    
    def get_model_hf_info(self, model_id: str) -> Dict[str, str]:
        """Get Hugging Face repository info for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with 'repo_id' and 'filename'
        """
        model_config = self.get_model_config(model_id)
        return {
            'repo_id': model_config.get('hf_repo_id', ''),
            'filename': model_config.get('filename', '')
        }
    
    def get_generation_settings(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get generation settings for a model.
        
        Args:
            model_id: Model identifier (uses defaults if not specified)
            
        Returns:
            Generation settings dictionary
        """
        defaults = self.model_config.get('generation_defaults', {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'max_tokens': 640
        })
        
        if model_id:
            model_config = self.get_model_config(model_id)
            # Model-specific settings override defaults
            if 'generation' in model_config:
                defaults.update(model_config['generation'])
        
        return defaults
    
    def get_models_for_ui(self) -> List[Dict[str, Any]]:
        """Get model information formatted for UI display.
        
        Returns:
            List of model info dictionaries for UI
        """
        models = self.get_available_models()
        ui_models = []
        
        for model_id, config in models.items():
            ui_models.append({
                'id': model_id,
                'name': config.get('display_name', model_id),
                'description': config.get('description', ''),
                'size': config.get('size_gb', 0),
                'parameters': config.get('parameters', ''),
                'tier': config.get('tier', 'balanced'),
                'default': config.get('default', False),
                'recommended_for': config.get('recommended_for', [])
            })
        
        # Sort by size (smallest first)
        ui_models.sort(key=lambda x: x['size'])
        
        return ui_models


# Global configuration loader instance
config_loader = ConfigLoader() 