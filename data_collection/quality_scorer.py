"""
Quality Scoring for Training Data

Automated assessment of training example quality to ensure
only high-quality data is used for fine-tuning.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class QualityMetrics:
    """Quality metrics for a training example"""
    overall_score: float
    consensus_quality: float
    research_depth: float
    reasoning_clarity: float
    response_completeness: float
    tool_usage_effectiveness: float
    issues: List[str]

class QualityScorer:
    """Automated quality assessment for training examples"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def score_example(self, example: Dict[str, Any]) -> QualityMetrics:
        """Score a single training example"""
        issues = []
        
        # Extract components
        prompt = example.get('prompt', '')
        final_output = example.get('final', '')
        tools = example.get('tools', [])
        metadata = example.get('meta', {})
        
        # Calculate individual scores
        consensus_quality = self._score_consensus_quality(final_output, metadata)
        research_depth = self._score_research_depth(tools, final_output)
        reasoning_clarity = self._score_reasoning_clarity(final_output)
        response_completeness = self._score_completeness(prompt, final_output)
        tool_usage = self._score_tool_usage(tools, final_output)
        
        # Calculate overall score
        scores = [
            consensus_quality,
            research_depth, 
            reasoning_clarity,
            response_completeness,
            tool_usage
        ]
        
        overall_score = np.mean(scores)
        
        # Identify issues
        if consensus_quality < 0.6:
            issues.append("Low consensus quality")
        if research_depth < 0.5:
            issues.append("Insufficient research depth")
        if reasoning_clarity < 0.7:
            issues.append("Unclear reasoning")
        if response_completeness < 0.8:
            issues.append("Incomplete response")
        if tool_usage < 0.6:
            issues.append("Poor tool utilization")
            
        return QualityMetrics(
            overall_score=overall_score,
            consensus_quality=consensus_quality,
            research_depth=research_depth,
            reasoning_clarity=reasoning_clarity,
            response_completeness=response_completeness,
            tool_usage_effectiveness=tool_usage,
            issues=issues
        )
    
    def _score_consensus_quality(self, output: str, metadata: Dict) -> float:
        """Score how well consensus was achieved"""
        score = 0.5  # Base score
        
        # Check for consensus indicators
        consensus_indicators = [
            'consensus', 'agreement', 'unanimous', 'collectively',
            'we agree', 'we conclude', 'we recommend', 'we find'
        ]
        
        for indicator in consensus_indicators:
            if indicator.lower() in output.lower():
                score += 0.1
                
        # Check metadata
        if metadata.get('consensus_achieved', False):
            score += 0.2
            
        return min(score, 1.0)
    
    def _score_research_depth(self, tools: List[Dict], output: str) -> float:
        """Score the depth and quality of research"""
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
        
        # Check for citations/references in output
        if any(word in output.lower() for word in ['according to', 'research shows', 'studies indicate']):
            score += 0.2
            
        return min(score, 1.0)
    
    def _score_reasoning_clarity(self, output: str) -> float:
        """Score the clarity and logical flow of reasoning"""
        score = 0.5  # Base score
        
        # Check for logical structure
        structure_indicators = [
            'first', 'second', 'third', 'finally',
            'however', 'therefore', 'consequently',
            'on the one hand', 'on the other hand'
        ]
        
        for indicator in structure_indicators:
            if indicator.lower() in output.lower():
                score += 0.05
                
        # Check for clear conclusions
        if any(word in output.lower() for word in ['conclusion', 'recommendation', 'therefore']):
            score += 0.2
            
        # Penalize very short responses
        if len(output.split()) < 50:
            score -= 0.3
            
        return max(min(score, 1.0), 0.0)
    
    def _score_completeness(self, prompt: str, output: str) -> float:
        """Score how completely the prompt was addressed"""
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
    
    def _score_tool_usage(self, tools: List[Dict], output: str) -> float:
        """Score how effectively tools were used"""
        if not tools:
            return 0.0
            
        score = 0.3  # Base score for having tools
        
        # Check if tool results are referenced in output
        tool_names = [tool.get('name', '') for tool in tools]
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
    
    def filter_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter examples based on quality threshold"""
        filtered = []
        
        for example in examples:
            metrics = self.score_example(example)
            if metrics.overall_score >= self.threshold:
                # Add quality metrics to example
                example['quality_metrics'] = {
                    'overall_score': metrics.overall_score,
                    'consensus_quality': metrics.consensus_quality,
                    'research_depth': metrics.research_depth,
                    'reasoning_clarity': metrics.reasoning_clarity,
                    'response_completeness': metrics.response_completeness,
                    'tool_usage_effectiveness': metrics.tool_usage_effectiveness,
                    'issues': metrics.issues
                }
                filtered.append(example)
                
        return filtered
