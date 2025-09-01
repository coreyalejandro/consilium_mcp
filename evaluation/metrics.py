"""
Consensus Metrics for Model Evaluation

Implements metrics specific to consensus quality and multi-AI collaboration.
"""

import re
from typing import Dict, List, Any
import numpy as np

class ConsensusMetrics:
    """Metrics for evaluating consensus quality"""
    
    def __init__(self):
        self.consensus_indicators = [
            'consensus', 'agreement', 'unanimous', 'collectively',
            'we agree', 'we conclude', 'we recommend', 'we find',
            'collective decision', 'shared understanding'
        ]
        
        self.structure_indicators = [
            'first', 'second', 'third', 'finally',
            'however', 'therefore', 'consequently',
            'on the one hand', 'on the other hand',
            'in conclusion', 'to summarize'
        ]
        
        self.research_indicators = [
            'according to', 'research shows', 'studies indicate',
            'evidence suggests', 'data shows', 'analysis reveals'
        ]
    
    def calculate_consensus_quality(self, output: str, metadata: Dict[str, Any]) -> float:
        """Calculate consensus quality score"""
        score = 0.5  # Base score
        
        # Check for consensus indicators
        for indicator in self.consensus_indicators:
            if indicator.lower() in output.lower():
                score += 0.1
                
        # Check metadata
        if metadata.get('consensus_achieved', False):
            score += 0.2
            
        # Check for collaborative language
        collaborative_words = ['we', 'our', 'together', 'collectively']
        for word in collaborative_words:
            if word.lower() in output.lower():
                score += 0.05
                
        return min(score, 1.0)
    
    def calculate_research_depth(self, tools: List[Dict[str, Any]], output: str) -> float:
        """Calculate research depth score"""
        if not tools:
            return 0.0
            
        score = 0.3  # Base score for having tools
        
        # Count different research sources
        source_types = set()
        for tool in tools:
            tool_name = tool.get('name', '')
            if 'web' in tool_name:
                source_types.add('web')
            elif 'arxiv' in tool_name:
                source_types.add('academic')
            elif 'github' in tool_name:
                source_types.add('code')
            elif 'sec' in tool_name:
                source_types.add('financial')
            elif 'wiki' in tool_name:
                source_types.add('reference')
                
        # More diverse sources = higher score
        score += min(len(source_types) * 0.15, 0.4)
        
        # Check for research citations
        for indicator in self.research_indicators:
            if indicator.lower() in output.lower():
                score += 0.1
                
        return min(score, 1.0)
    
    def calculate_reasoning_clarity(self, output: str) -> float:
        """Calculate reasoning clarity score"""
        score = 0.5  # Base score
        
        # Check for logical structure
        for indicator in self.structure_indicators:
            if indicator.lower() in output.lower():
                score += 0.05
                
        # Check for clear conclusions
        conclusion_indicators = ['conclusion', 'recommendation', 'therefore', 'thus']
        for indicator in conclusion_indicators:
            if indicator.lower() in output.lower():
                score += 0.1
                
        # Penalize very short responses
        if len(output.split()) < 50:
            score -= 0.3
            
        # Bonus for well-structured responses
        if len(output.split()) > 200:
            score += 0.1
            
        return max(min(score, 1.0), 0.0)
    
    def calculate_completeness(self, prompt: str, output: str) -> float:
        """Calculate response completeness score"""
        score = 0.5  # Base score
        
        # Check if output addresses the prompt
        prompt_words = set(prompt.lower().split())
        output_words = set(output.lower().split())
        
        # Simple word overlap
        overlap = len(prompt_words.intersection(output_words))
        if len(prompt_words) > 0:
            overlap_ratio = overlap / len(prompt_words)
            score += overlap_ratio * 0.3
            
        # Check for comprehensive response
        if len(output.split()) > 200:
            score += 0.2
            
        return min(score, 1.0)
    
    def calculate_tool_usage(self, tools: List[Dict[str, Any]], output: str) -> float:
        """Calculate tool usage effectiveness score"""
        if not tools:
            return 0.0
            
        score = 0.3  # Base score for having tools
        
        # Check if tool results are referenced in output
        tool_content = [tool.get('content', '') for tool in tools]
        
        # Look for tool content being used in output
        for content in tool_content:
            if len(content) > 50:  # Substantial tool result
                # Simple check for content overlap
                content_words = set(content.lower().split()[:20])  # First 20 words
                output_words = set(output.lower().split())
                if len(content_words.intersection(output_words)) > 2:
                    score += 0.2
                    break
                    
        return min(score, 1.0)
    
    def calculate_consensus_strength(self, output: str) -> float:
        """Calculate the strength of consensus achieved"""
        score = 0.0
        
        # Strong consensus indicators
        strong_indicators = ['unanimous', 'complete agreement', 'full consensus']
        for indicator in strong_indicators:
            if indicator.lower() in output.lower():
                score += 0.3
                
        # Moderate consensus indicators
        moderate_indicators = ['general agreement', 'broad consensus', 'majority agree']
        for indicator in moderate_indicators:
            if indicator.lower() in output.lower():
                score += 0.2
                
        # Weak consensus indicators
        weak_indicators = ['some agreement', 'partial consensus', 'mixed views']
        for indicator in weak_indicators:
            if indicator.lower() in output.lower():
                score += 0.1
                
        return min(score, 1.0)
    
    def calculate_controversy_level(self, output: str) -> float:
        """Calculate the level of controversy in the response"""
        controversy_indicators = [
            'disagreement', 'controversy', 'debate', 'conflict',
            'differing views', 'opposing opinions', 'dispute'
        ]
        
        score = 0.0
        for indicator in controversy_indicators:
            if indicator.lower() in output.lower():
                score += 0.2
                
        return min(score, 1.0)
