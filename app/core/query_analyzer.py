"""
Config-driven query analysis for Meetara Core.
Replaces hardcoded keywords and domains with dynamic config-based analysis.
"""
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from pathlib import Path
from app.core.logger import agent_logger
from app.utils.yaml_loader import load_yaml_file  # ✅ Use centralized YAML loader


class QueryAnalyzer:
    """Config-driven query analysis using domain and keyword configurations."""
    
    def __init__(self):
        self.domain_config = self._load_domain_config()
        self.domain_keywords = self._load_domain_keywords()
        self.intent_patterns = self._load_intent_patterns()
        self.generic_terms = self._load_generic_terms()
        self.tier_config = self._load_tier_config()
        self.context_rules = self._load_context_rules()  # New: Load context rules
        
        # Build cross-domain intelligence
        self.keyword_uniqueness = self._build_keyword_uniqueness_map()
        
        agent_logger.info("QueryAnalyzer initialized with config-driven patterns and cross-domain intelligence")
    
    def _load_domain_config(self) -> Dict[str, Any]:
        """Load domain configuration from YAML."""
        config_path = Path("config/domain_config.yaml")
        loaded = load_yaml_file(config_path, default=None)
        if loaded is not None:
            return loaded
        else:
            agent_logger.warning("domain_config.yaml not found, using default config")
            return self._get_default_domain_config()
    
    def _load_domain_keywords(self) -> Dict[str, Any]:
        """Load domain keywords from YAML."""
        config_path = Path("config/domain_keywords.yaml")
        loaded = load_yaml_file(config_path, default=None)
        if loaded is not None:
            return loaded
        else:
            agent_logger.warning("domain_keywords.yaml not found, using default keywords")
            return self._get_default_domain_keywords()
    
    def _load_generic_terms(self) -> List[str]:
        """Load generic terms from domain_keywords.yaml."""
        try:
            generic_terms = self.domain_keywords.get("generic_terms", [])
            if generic_terms:
                agent_logger.info(f"Loaded {len(generic_terms)} generic terms from config")
                return [term.lower() for term in generic_terms]
            else:
                agent_logger.warning("No generic_terms found in domain_keywords.yaml, using defaults")
                return ["techniques", "methods", "strategies", "tips", "help", "support", "approach", "ways"]
        except Exception as e:
            agent_logger.error(f"Error loading generic terms: {e}")
            return ["techniques", "methods", "strategies", "tips", "help", "support", "approach", "ways"]
    
    def _load_tier_config(self) -> Dict[str, Any]:
        """Load tier configuration from YAML."""
        config_path = Path("config/tier_config.yaml")
        loaded = load_yaml_file(config_path, default=None)
        if loaded is not None:
            return loaded
        else:
            agent_logger.warning("tier_config.yaml not found, using default tier weights")
            return {"tiers": {}}
    
    def _load_context_rules(self) -> Dict[str, Dict[str, Set[str]]]:
        """Load context rules from domain_keywords.yaml."""
        try:
            context_rules_raw = self.domain_keywords.get("context_rules", {})
            context_rules = {}
            
            for domain, rules in context_rules_raw.items():
                context_rules[domain] = {
                    "context_dependent": set(kw.lower() for kw in rules.get("context_dependent", [])),
                    "context_indicators": set(kw.lower() for kw in rules.get("context_indicators", []))
                }
            
            if context_rules:
                agent_logger.info(f"✅ Loaded context rules for {len(context_rules)} domains")
            return context_rules
        except Exception as e:
            agent_logger.warning(f"Failed to load context rules: {e}")
            return {}
    
    def _generate_ngrams(self, text: str, max_n: int = 4) -> Dict[int, Set[str]]:
        """Generate n-grams (1-word, 2-word, 3-word, 4-word phrases) from text.
        
        Example: "improve my mental wellbeing"
        1-grams: {"improve", "my", "mental", "wellbeing"}
        2-grams: {"improve my", "my mental", "mental wellbeing"}
        3-grams: {"improve my mental", "my mental wellbeing"}
        4-grams: {"improve my mental wellbeing"}
        """
        words = text.split()
        ngrams = {}
        
        for n in range(1, min(max_n + 1, len(words) + 1)):
            ngrams[n] = set()
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngrams[n].add(ngram)
        
        return ngrams
    
    def _build_keyword_uniqueness_map(self) -> Dict[str, int]:
        """Build a map of how many domains each keyword appears in (cross-domain analysis)."""
        keyword_counts = {}
        
        domains_data = self.domain_keywords.get("domains", {})
        for domain_name, domain_data in domains_data.items():
            keywords = domain_data.get("keywords", [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1
        
        agent_logger.info(f"Built keyword uniqueness map: {len(keyword_counts)} unique keywords across {len(domains_data)} domains")
        
        # Log some stats
        unique_keywords = sum(1 for count in keyword_counts.values() if count == 1)
        common_keywords = sum(1 for count in keyword_counts.values() if count >= 4)
        agent_logger.info(f"  - Unique keywords (1 domain): {unique_keywords}")
        agent_logger.info(f"  - Common keywords (4+ domains): {common_keywords}")
        
        return keyword_counts
    
    def get_top_candidate_domains(
        self,
        query: str,
        top_n: int = 5,
        context_domain: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top N candidate domains with scores (for semantic search hints).
        Returns list of (domain, score) tuples sorted by score descending.
        Even returns low-score domains to give semantic search a chance.
        """
        query_lower = query.lower()
        domain_scores = {}
        
        # Check context domain first
        if context_domain:
            domain_scores[context_domain] = 0.5
        
        # Generate n-grams from query
        query_ngrams = self._generate_ngrams(query_lower, max_n=4)
        
        # Score all domains
        for domain_name, domain_data in self.domain_keywords.get("domains", {}).items():
            keywords = domain_data.get("keywords", [])
            score = 0
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_word_count = len(keyword_lower.split())
                
                matched = False
                if keyword_word_count <= 4 and keyword_word_count in query_ngrams:
                    matched = keyword_lower in query_ngrams[keyword_word_count]
                else:
                    matched = keyword_lower in query_lower
                
                if matched:
                    word_count = len(keyword_lower.split())
                    if word_count >= 3:
                        base_score = 15
                    elif word_count == 2:
                        base_score = 8
                    else:
                        base_score = 1
                    
                    domain_count = self.keyword_uniqueness.get(keyword_lower, 1)
                    if domain_count == 1:
                        uniqueness_bonus = 5
                    elif domain_count <= 3:
                        uniqueness_bonus = 2
                    else:
                        uniqueness_bonus = 0
                    
                    if keyword_lower in self.generic_terms:
                        generic_penalty = -1
                    else:
                        generic_penalty = 0
                    
                    keyword_score = base_score + uniqueness_bonus + generic_penalty
                    final_score = max(keyword_score, 0)
                    
                    if final_score > 0:
                        score += final_score
            
            tier_multiplier = self._get_tier_multiplier(domain_name)
            score = score * tier_multiplier
            
            if score > 0:
                domain_scores[domain_name] = domain_scores.get(domain_name, 0) + score
        
        # Return top N domains sorted by score (even if scores are low)
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_domains[:top_n]
    
    def _load_intent_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load intent patterns for query enhancement."""
        return {
            "symptoms": {
                "keywords": ["symptoms", "signs", "warning", "indication", "early warning"],
                "enhancement": ["symptoms", "signs", "indicators", "warning signs", "early detection"]
            },
            "treatment": {
                "keywords": ["treatment", "therapy", "medication", "cure", "remedy"],
                "enhancement": ["treatment", "therapy", "medication", "management", "options"]
            },
            "causes": {
                "keywords": ["causes", "risk", "prevention", "why", "reason"],
                "enhancement": ["causes", "risk factors", "prevention", "triggers", "contributing factors"]
            },
            "diagnosis": {
                "keywords": ["diagnosis", "test", "screening", "detection", "check"],
                "enhancement": ["diagnosis", "testing", "screening", "detection", "evaluation"]
            },
            "prevention": {
                "keywords": ["prevent", "avoid", "protection", "safety", "prevention"],
                "enhancement": ["prevention", "protection", "avoidance", "safety measures", "risk reduction"]
            }
        }
    
    def _get_default_domain_config(self) -> Dict[str, Any]:
        """Default domain configuration if config file is not available."""
        return {
            "categories": {
                "healthcare": {
                    "domains": {
                        "women_health": {"priority": 1},
                        "general_health": {"priority": 1},
                        "mental_health": {"priority": 1}
                    }
                }
            }
        }
    
    def _get_default_domain_keywords(self) -> Dict[str, Any]:
        """Default domain keywords if config file is not available."""
        return {
            "domains": {
                "women_health": {
                    "keywords": ["women's health", "gynecology", "pregnancy", "menstruation", "fertility", "breast health", "reproductive health", "cervical", "ovarian", "uterine"]
                },
                "general_health": {
                    "keywords": ["health", "medical", "doctor", "symptoms", "diagnosis", "treatment", "medicine", "wellness", "prevention"]
                },
                "mental_health": {
                    "keywords": ["mental health", "psychology", "therapy", "counseling", "depression", "anxiety", "stress", "emotions", "mindfulness"]
                }
            }
        }
    
    def analyze_query(self, query: str, context_domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze query to determine domain, intent, and enhancement keywords.
        
        Args:
            query: User query string
            context_domain: Optional domain from context
            
        Returns:
            Dict containing:
            - detected_domain: Best matching domain
            - detected_intent: Query intent (symptoms, treatment, etc.)
            - enhanced_query: Query with relevant keywords added
            - confidence: Confidence score (0-1)
        """
        query_lower = query.lower()
        
        # Detect domain (now returns domain and score)
        detected_domain, domain_score = self._detect_domain(query_lower, context_domain)
        
        # Detect intent
        detected_intent = self._detect_intent(query_lower)
        
        # Enhance query
        enhanced_query = self._enhance_query(query, detected_intent, detected_domain)
        
        # Calculate confidence based on domain score
        confidence = self._calculate_confidence_from_score(domain_score, detected_intent)
        
        return {
            "detected_domain": detected_domain,
            "detected_intent": detected_intent,
            "enhanced_query": enhanced_query,
            "confidence": confidence,
            "original_query": query
        }
    
    def _detect_domain(self, query_lower: str, context_domain: Optional[str] = None) -> Tuple[str, float]:
        """Detect the most relevant domain with SMART CROSS-DOMAIN scoring."""
        domain_scores = {}
        
        # Check context domain first
        if context_domain:
            domain_scores[context_domain] = 0.5  # Base score for context domain
        
        # Generate n-grams from query for smart phrase matching
        query_ngrams = self._generate_ngrams(query_lower, max_n=4)
        
        # Score all domains based on keyword matches with INTELLIGENT CROSS-DOMAIN scoring
        for domain_name, domain_data in self.domain_keywords.get("domains", {}).items():
            keywords = domain_data.get("keywords", [])
            score = 0
            matched_keywords = []  # Track what matched for debugging
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_word_count = len(keyword_lower.split())
                
                # Check if keyword exists in appropriate n-gram set
                matched = False
                if keyword_word_count <= 4 and keyword_word_count in query_ngrams:
                    matched = keyword_lower in query_ngrams[keyword_word_count]
                else:
                    # Fallback for long phrases
                    matched = keyword_lower in query_lower
                
                if matched:
                    # === CONTEXT-AWARE MATCHING (Config-Driven) ===
                    
                    # Check if this domain has context rules
                    if domain_name in self.context_rules:
                        rules = self.context_rules[domain_name]
                        context_dependent_keywords = rules.get("context_dependent", set())
                        context_indicators = rules.get("context_indicators", set())
                        
                        # If keyword requires context, check for indicators
                        if keyword_lower in context_dependent_keywords:
                            has_context = any(
                                indicator in query_lower 
                                for indicator in context_indicators
                            )
                            if not has_context:
                                continue  # Skip this keyword - no context found!
                    
                    # === SMART SCORING ===
                    
                    # 1. Base score by word count (BOOSTED for multi-word phrases!)
                    word_count = len(keyword_lower.split())
                    if word_count >= 3:
                        base_score = 15  # "mental health support" - VERY specific!
                    elif word_count == 2:
                        base_score = 8   # "mental health" - specific
                    else:
                        base_score = 1   # "anxiety" - single word
                    
                    # 2. Uniqueness bonus (cross-domain intelligence!)
                    domain_count = self.keyword_uniqueness.get(keyword_lower, 1)
                    if domain_count == 1:
                        uniqueness_bonus = 5  # Very unique (e.g., "ptsd", "ocd")
                    elif domain_count <= 3:
                        uniqueness_bonus = 2  # Somewhat unique (e.g., "anxiety")
                    else:
                        uniqueness_bonus = 0  # Common word (e.g., "health", "management")
                    
                    # 3. Generic term penalty
                    if keyword_lower in self.generic_terms:
                        generic_penalty = -1  # Reduce score for generic terms
                    else:
                        generic_penalty = 0
                    
                    # Calculate total score for this keyword
                    keyword_score = base_score + uniqueness_bonus + generic_penalty
                    final_score = max(keyword_score, 0)  # Never negative
                    
                    if final_score > 0:
                        score += final_score
                        matched_keywords.append(f"{keyword_lower}({final_score})")
            
            # 4. Apply tier-based multiplier (safety_critical > expert > quality)
            tier_multiplier = self._get_tier_multiplier(domain_name)
            score = score * tier_multiplier
            
            if score > 0:
                domain_scores[domain_name] = domain_scores.get(domain_name, 0) + score
                # Log what keywords matched for debugging
                if len(matched_keywords) <= 5:  # Only log if reasonable number
                    agent_logger.debug(f"   {domain_name}: {', '.join(matched_keywords[:5])} = {score:.1f}")
        
        # Return the domain with highest score, or default
        if domain_scores:
            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            best_domain = sorted_domains[0][0]
            best_score = sorted_domains[0][1]
            top_3 = sorted_domains[:3]
            agent_logger.info(f"✅ Detected domain: {best_domain} (score: {best_score:.1f})")
            agent_logger.info(f"   Top 3: {[(d, f'{s:.1f}') for d, s in top_3]}")
            
            # If best score is very low (< 1.0), log warning but still return it
            if best_score < 1.0:
                agent_logger.warning(f"⚠️ Low confidence domain detection: {best_domain} (score: {best_score:.1f})")
            
            return best_domain, best_score
        
        # Fallback: Try to detect from query content if no keywords matched
        # Check for obvious math/academic terms that might not be in keywords
        math_terms = ["calculus", "theorem", "derivative", "integral", "algebra", "geometry", "trigonometry", 
                     "equation", "solve", "formula", "proof", "mathematical", "mathematics", "math"]
        if any(term in query_lower for term in math_terms):
            agent_logger.info(f"🔍 No keyword matches, but detected math terms - using academic_tutoring as fallback")
            return "academic_tutoring", 0.5
        
        # Fallback to context domain or default
        fallback_domain = context_domain or "general_health"
        agent_logger.warning(f"⚠️ No domain matches found, using fallback: {fallback_domain}")
        return fallback_domain, 0.0
    
    def _get_tier_multiplier(self, domain_name: str) -> float:
        """Get tier-based multiplier for domain scoring."""
        # Find which tier this domain belongs to
        tiers = self.tier_config.get("tiers", {})
        
        for tier_name, tier_data in tiers.items():
            tier_domains = tier_data.get("domains", [])
            if domain_name in tier_domains:
                # Tier multipliers
                if tier_name == "safety_critical":
                    return 1.3  # Healthcare, emergency - highest priority
                elif tier_name == "expert":
                    return 1.1  # Business, tech, education - high priority
                elif tier_name == "quality":
                    return 1.0  # Creative, lifestyle - normal priority
        
        # Default multiplier
        return 1.0
    
    def _detect_intent(self, query_lower: str) -> Optional[str]:
        """Detect the intent of the query (symptoms, treatment, causes, etc.)."""
        intent_scores = {}
        
        for intent, pattern_data in self.intent_patterns.items():
            keywords = pattern_data["keywords"]
            score = 0
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            agent_logger.debug(f"Detected intent: {best_intent} (score: {intent_scores[best_intent]})")
            return best_intent
        
        return None
    
    def _enhance_query(self, original_query: str, intent: Optional[str], domain: str) -> str:
        """Enhance query with relevant keywords based on intent and domain."""
        enhanced_parts = [original_query]
        
        # Add intent-based enhancement
        if intent and intent in self.intent_patterns:
            enhancement_keywords = self.intent_patterns[intent]["enhancement"]
            enhanced_parts.extend(enhancement_keywords[:3])  # Add top 3 enhancement keywords
        
        # Add domain-specific enhancement
        domain_data = self.domain_keywords.get("domains", {}).get(domain, {})
        domain_keywords = domain_data.get("keywords", [])
        
        # Add relevant domain keywords (avoid duplicates)
        added_keywords = set()
        for keyword in domain_keywords[:5]:  # Add top 5 domain keywords
            if keyword.lower() not in original_query.lower() and len(added_keywords) < 3:
                added_keywords.add(keyword)
        
        enhanced_parts.extend(list(added_keywords))
        
        enhanced_query = " ".join(enhanced_parts)
        agent_logger.debug(f"Enhanced query: {enhanced_query}")
        return enhanced_query
    
    def _calculate_confidence_from_score(self, domain_score: float, intent: Optional[str]) -> float:
        """Calculate confidence score based on domain detection score."""
        confidence = 0.0
        
        # Normalize domain score to 0-1 range (domain scores are typically 1-100)
        # Typical scores: 1-10 (low), 10-30 (medium), 30+ (high)
        if domain_score >= 30:
            domain_confidence = 1.0
        elif domain_score >= 15:
            domain_confidence = 0.8
        elif domain_score >= 5:
            domain_confidence = 0.6
        elif domain_score >= 1:
            domain_confidence = 0.4
        else:
            domain_confidence = 0.2
        
        confidence += domain_confidence * 0.6  # Domain is 60% of confidence
        
        # Intent confidence (currently not used much, but available)
        if intent:
            confidence += 0.2  # Add some confidence if intent detected
        
        return min(confidence, 1.0)
    
    def _calculate_confidence(self, query_lower: str, domain: str, intent: Optional[str]) -> float:
        """OLD METHOD: Calculate confidence score for the analysis based on SMART keyword matches using n-grams.
        DEPRECATED: Use _calculate_confidence_from_score instead."""
        confidence = 0.0
        
        # Generate n-grams from query for smart phrase matching (same as _detect_domain)
        query_ngrams = self._generate_ngrams(query_lower, max_n=4)
        
        # Domain confidence - use n-gram matching like _detect_domain
        domain_data = self.domain_keywords.get("domains", {}).get(domain, {})
        domain_keywords = domain_data.get("keywords", [])
        
        domain_matches = 0
        for keyword in domain_keywords:
            keyword_lower = keyword.lower()
            keyword_word_count = len(keyword_lower.split())
            
            # Check if keyword exists in appropriate n-gram set
            matched = False
            if keyword_word_count <= 4 and keyword_word_count in query_ngrams:
                matched = keyword_lower in query_ngrams[keyword_word_count]
            else:
                # Fallback for long phrases
                matched = keyword_lower in query_lower
            
            # Apply context-aware matching if needed
            if matched and domain in self.context_rules:
                rules = self.context_rules[domain]
                context_dependent_keywords = rules.get("context_dependent", set())
                context_indicators = rules.get("context_indicators", set())
                
                if keyword_lower in context_dependent_keywords:
                    has_context = any(indicator in query_lower for indicator in context_indicators)
                    if not has_context:
                        matched = False  # Skip this keyword - no context found!
            
            if matched:
                domain_matches += 1
        
        # Score based on absolute matches:
        # 1 match = 0.3, 2 matches = 0.5, 3+ matches = 0.7-1.0
        if domain_matches >= 3:
            domain_confidence = min(0.7 + (domain_matches - 3) * 0.1, 1.0)
        elif domain_matches == 2:
            domain_confidence = 0.5
        elif domain_matches == 1:
            domain_confidence = 0.3
        else:
            domain_confidence = 0.0
        
        confidence += domain_confidence * 0.6  # Domain is 60% of confidence
        
        # Intent confidence
        if intent:
            intent_data = self.intent_patterns.get(intent, {})
            intent_keywords = intent_data.get("keywords", [])
            
            intent_matches = sum(1 for keyword in intent_keywords if keyword.lower() in query_lower)
            if intent_keywords:
                intent_confidence = min(intent_matches / len(intent_keywords), 1.0)
                confidence += intent_confidence * 0.4  # Intent is 40% of confidence
        
        return min(confidence, 1.0)
    
    def get_response_template(self, domain: str, intent: Optional[str]) -> Dict[str, str]:
        """Get response template based on domain and intent."""
        templates = {
            "women_health": {
                "symptoms": {
                    "intro": "Based on our women's health knowledge base, here are some common symptoms that may be associated with this condition.",
                    "disclaimer": "⚠️ **Important Medical Disclaimer:**\n• These symptoms can also be caused by many other conditions\n• Only a qualified healthcare provider can make a proper diagnosis\n• If you have any concerns, please consult with a healthcare professional immediately",
                    "header": "**Common Symptoms:**"
                },
                "treatment": {
                    "intro": "Based on our women's health knowledge base, here are some common treatment options.",
                    "disclaimer": "⚠️ **Important Medical Disclaimer:**\n• Treatment plans are highly individualized\n• All treatment decisions should be discussed with your healthcare team\n• This information is for general education only",
                    "header": "**Treatment Information:**"
                },
                "default": {
                    "intro": "Here's what I found about this topic in our women's health knowledge base.",
                    "disclaimer": "⚠️ **Important Medical Disclaimer:**\n• This information is general in nature\n• Every person's situation is unique\n• Always consult healthcare professionals for specific medical advice",
                    "header": "**Women's Health Information:**"
                }
            },
            "general_health": {
                "default": {
                    "intro": "Based on your question, here's what I found in our medical knowledge base.",
                    "disclaimer": "⚠️ **Important Medical Disclaimer:**\n• This information is general in nature\n• Always consult qualified healthcare professionals\n• Medical decisions require professional evaluation",
                    "header": "**Medical Information:**"
                }
            }
        }
        
        # ✅ FIX: Don't default to medical template for unknown domains
        # Get domain templates - use generic default instead of medical
        domain_templates = templates.get(domain, {})
        
        # Generic default template (domain-agnostic)
        generic_default = {
            "intro": "Based on your question, here's what I found in our knowledge base.",
            "disclaimer": "",  # No disclaimer for generic domains
            "header": "**Information:**"
        }
        
        # Get intent-specific template or default
        if intent and intent in domain_templates:
            return domain_templates[intent]
        else:
            # ✅ Only use medical template if domain is actually medical
            if domain.lower() in ["general_health", "health", "medical", "mental_health", "women_health"]:
                return domain_templates.get("default", templates.get("general_health", {}).get("default", generic_default))
            else:
                # For non-medical domains, use generic template
                return domain_templates.get("default", generic_default)
    
    def filter_relevant_content(self, content: str, query_lower: str, domain: str, intent: Optional[str]) -> List[str]:
        """Filter and clean content based on domain and intent."""
        relevant_sentences = []
        
        # Get domain keywords for filtering
        domain_data = self.domain_keywords.get("domains", {}).get(domain, {})
        domain_keywords = [kw.lower() for kw in domain_data.get("keywords", [])]
        
        # Get intent keywords for filtering
        intent_keywords = []
        if intent and intent in self.intent_patterns:
            intent_keywords = [kw.lower() for kw in self.intent_patterns[intent]["keywords"]]
        
        # Split content into sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Skip sentences that are too short or contain unwanted content
            if len(sentence) < 10 or sentence.isdigit() or \
               any(term in sentence_lower for term in ["http://", "www.", "800-", ".gov", ".com", ".org"]):
                continue
            
            # Clean the sentence
            cleaned_sentence = self._clean_sentence(sentence)
            if len(cleaned_sentence) < 10:
                continue
            
            # Check relevance based on domain and intent
            is_relevant = False
            
            # Check domain relevance
            if any(keyword in sentence_lower for keyword in domain_keywords):
                is_relevant = True
            
            # Check intent relevance
            if intent and any(keyword in sentence_lower for keyword in intent_keywords):
                is_relevant = True
            
            # If no specific intent, check general relevance
            if not intent and any(keyword in sentence_lower for keyword in query_lower.split()):
                is_relevant = True
            
            if is_relevant:
                relevant_sentences.append(f"• {cleaned_sentence}")
        
        return relevant_sentences
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and format a sentence."""
        # Remove leading numbers and dashes
        cleaned = re.sub(r'^\d+[\s\.\-]+', '', sentence)
        # Remove leading bullet points and dashes
        cleaned = re.sub(r'^[•\-\*\+]+\s*', '', cleaned)
        # Remove any remaining leading/trailing whitespace
        cleaned = cleaned.strip()
        return cleaned


# Global instance
_query_analyzer: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """Get the global QueryAnalyzer instance."""
    global _query_analyzer
    if _query_analyzer is None:
        _query_analyzer = QueryAnalyzer()
    return _query_analyzer 