"""
Base class for all research tools
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import re
from datetime import datetime


class BaseTool(ABC):
    """Base class for all research tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # seconds between requests
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> str:
        """Main search method - must be implemented by subclasses"""
        pass
    
    def rate_limit(self):
        """Simple rate limiting to be respectful to APIs"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def score_research_quality(self, research_result: str, source: str = "web") -> Dict[str, float]:
        """Score research based on multiple quality indicators"""
        
        quality_score = {
            "recency": self._check_recency(research_result),
            "authority": self._check_authority(research_result, source),
            "specificity": self._check_specificity(research_result),
            "relevance": self._check_relevance(research_result),
            "overall": 0.0
        }
        
        # Weighted overall score
        weights = {"recency": 0.2, "authority": 0.3, "specificity": 0.3, "relevance": 0.2}
        quality_score["overall"] = sum(quality_score[metric] * weight for metric, weight in weights.items())
        
        return quality_score
    
    def _check_recency(self, text: str) -> float:
        """Check for recent dates and current information"""
        if not text:
            return 0.3
            
        # Look for years
        years = re.findall(r'\b(20\d{2})\b', text)
        if years:
            latest_year = max(int(year) for year in years)
            current_year = datetime.now().year
            recency = max(0, 1 - (current_year - latest_year) / 10)  # Decay over 10 years
            return recency
        return 0.3  # Default for no date found
    
    def _check_authority(self, text: str, source: str) -> float:
        """Check source authority and credibility indicators"""
        authority_indicators = {
            'arxiv': 0.9,
            'sec': 0.95,
            'github': 0.7,
            'wikipedia': 0.8,
            'web': 0.5
        }
        
        base_score = authority_indicators.get(source.lower(), 0.5)
        
        # Look for credibility markers in text
        if text:
            credibility_markers = ['study', 'research', 'university', 'published', 'peer-reviewed', 'official']
            marker_count = sum(1 for marker in credibility_markers if marker in text.lower())
            credibility_boost = min(0.3, marker_count * 0.05)
            base_score += credibility_boost
        
        return min(1.0, base_score)
    
    def _check_specificity(self, text: str) -> float:
        """Check for specific data points and quantitative information"""
        if not text:
            return 0.1
            
        # Count numbers, percentages, specific metrics
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
        specific_terms = len(re.findall(r'\b(?:exactly|precisely|specifically|measured|calculated)\b', text, re.IGNORECASE))
        
        specificity = min(1.0, (numbers * 0.02) + (specific_terms * 0.1))
        return max(0.1, specificity)  # Minimum baseline
    
    def _check_relevance(self, text: str) -> float:
        """Check relevance to query (simplified implementation)"""
        # This would ideally use the original query for comparison
        # For now, return a baseline that could be enhanced
        return 0.7  # Placeholder - could be enhanced with query matching
    
    def should_use_for_query(self, query: str) -> bool:
        """Determine if this tool should be used for the given query"""
        # Default implementation - override in subclasses for smart routing
        return True
    
    def extract_key_info(self, text: str) -> Dict[str, Any]:
        """Extract key information from research results"""
        if not text:
            return {}
            
        return {
            'length': len(text),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_dates': bool(re.search(r'\b20\d{2}\b', text)),
            'has_urls': bool(re.search(r'http[s]?://', text))
        }
    
    def format_error_response(self, query: str, error: str) -> str:
        """Format a consistent error response"""
        return f"**{self.name} Research for: {query}**\n\nResearch temporarily unavailable: {str(error)[:100]}..."