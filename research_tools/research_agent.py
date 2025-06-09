"""
Enhanced Research Agent with Multi-Source Integration
"""
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter

from .base_tool import BaseTool
from .web_search import WebSearchTool
from .wikipedia_search import WikipediaSearchTool
from .arxiv_search import ArxivSearchTool
from .github_search import GitHubSearchTool
from .sec_search import SECSearchTool


class EnhancedResearchAgent:
    """Enhanced research agent with multi-source synthesis and smart routing"""
    
    def __init__(self):
        # Initialize all research tools
        self.tools = {
            'web': WebSearchTool(),
            'wikipedia': WikipediaSearchTool(),
            'arxiv': ArxivSearchTool(),
            'github': GitHubSearchTool(),
            'sec': SECSearchTool()
        }
        
        # Tool availability status
        self.tool_status = {name: True for name in self.tools.keys()}
        
    def search(self, query: str, research_depth: str = "standard") -> str:
        """Main search method with intelligent routing"""
        if research_depth == "deep":
            return self._deep_multi_source_search(query)
        else:
            return self._standard_search(query)
    
    def search_wikipedia(self, topic: str) -> str:
        """Wikipedia search method for backward compatibility"""
        return self.tools['wikipedia'].search(topic)
    
    def _standard_search(self, query: str) -> str:
        """Standard single-source search with smart routing"""
        # Determine best tool for the query
        best_tool = self._route_query_to_tool(query)
        
        try:
            return self.tools[best_tool].search(query)
        except Exception as e:
            # Fallback to web search
            if best_tool != 'web':
                try:
                    return self.tools['web'].search(query)
                except Exception as e2:
                    return f"**Research for: {query}**\n\nResearch temporarily unavailable: {str(e2)[:100]}..."
            else:
                return f"**Research for: {query}**\n\nResearch temporarily unavailable: {str(e)[:100]}..."
    
    def _deep_multi_source_search(self, query: str) -> str:
        """Deep research using multiple sources with synthesis"""
        results = {}
        quality_scores = {}
        
        # Determine which sources to use based on query type
        relevant_tools = self._get_relevant_tools(query)
        
        # Collect results from multiple sources
        for tool_name in relevant_tools:
            try:
                result = self.tools[tool_name].search(query)
                if result and len(result.strip()) > 50:  # Ensure meaningful result
                    results[tool_name] = result
                    quality_scores[tool_name] = self.tools[tool_name].score_research_quality(result, tool_name)
            except Exception as e:
                print(f"Error with {tool_name}: {e}")
                continue
        
        if not results:
            return f"**Deep Research for: {query}**\n\nNo sources were able to provide results. Please try a different query."
        
        # Synthesize results
        return self._synthesize_multi_source_results(query, results, quality_scores)
    
    def _route_query_to_tool(self, query: str) -> str:
        """Intelligently route query to the most appropriate tool"""
        query_lower = query.lower()
        
        # Priority routing based on query characteristics
        for tool_name, tool in self.tools.items():
            if tool.should_use_for_query(query):
                # Return first matching tool based on priority order
                priority_order = ['arxiv', 'sec', 'github', 'wikipedia', 'web']
                if tool_name in priority_order[:3]:  # High-priority specialized tools
                    return tool_name
        
        # Secondary check for explicit indicators
        if any(indicator in query_lower for indicator in ['company', 'stock', 'financial', 'revenue']):
            return 'sec'
        elif any(indicator in query_lower for indicator in ['research', 'study', 'academic', 'paper']):
            return 'arxiv'
        elif any(indicator in query_lower for indicator in ['technology', 'framework', 'programming']):
            return 'github'
        elif any(indicator in query_lower for indicator in ['what is', 'definition', 'history']):
            return 'wikipedia'
        else:
            return 'web'  # Default fallback
    
    def _get_relevant_tools(self, query: str) -> List[str]:
        """Get list of relevant tools for deep search"""
        relevant_tools = []
        
        # Always include web search for current information
        relevant_tools.append('web')
        
        # Add specialized tools based on query
        for tool_name, tool in self.tools.items():
            if tool_name != 'web' and tool.should_use_for_query(query):
                relevant_tools.append(tool_name)
        
        # Ensure we don't overwhelm with too many sources
        if len(relevant_tools) > 4:
            # Prioritize specialized tools
            priority_order = ['arxiv', 'sec', 'github', 'wikipedia', 'web']
            relevant_tools = [tool for tool in priority_order if tool in relevant_tools][:4]
        
        return relevant_tools
    
    def _synthesize_multi_source_results(self, query: str, results: Dict[str, str], quality_scores: Dict[str, Dict]) -> str:
        """Synthesize results from multiple research sources"""
        synthesis = f"**Comprehensive Research Analysis: {query}**\n\n"
        
        # Add source summary
        synthesis += f"**Research Sources Used:** {', '.join(results.keys()).replace('_', ' ').title()}\n\n"
        
        # Find key themes and agreements/disagreements
        key_findings = self._extract_key_findings(results)
        synthesis += self._format_key_findings(key_findings)
        
        # Add individual source results (condensed)
        synthesis += "**Detailed Source Results:**\n\n"
        
        # Sort sources by quality score
        sorted_sources = sorted(quality_scores.items(), key=lambda x: x[1]['overall'], reverse=True)
        
        for source_name, _ in sorted_sources:
            if source_name in results:
                source_result = results[source_name]
                quality = quality_scores[source_name]
                
                # Condense long results
                if len(source_result) > 800:
                    source_result = source_result[:800] + "...\n[Result truncated for synthesis]"
                
                synthesis += f"**{source_name.replace('_', ' ').title()} (Quality: {quality['overall']:.2f}/1.0):**\n"
                synthesis += f"{source_result}\n\n"
        
        # Add research quality assessment
        synthesis += self._format_research_quality_assessment(quality_scores)
        
        return synthesis
    
    def _extract_key_findings(self, results: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract key findings and themes from multiple sources"""
        findings = {
            'agreements': [],
            'contradictions': [],
            'unique_insights': [],
            'data_points': []
        }
        
        # Extract key sentences from each source
        all_sentences = []
        source_sentences = {}
        
        for source, result in results.items():
            sentences = self._extract_key_sentences(result)
            source_sentences[source] = sentences
            all_sentences.extend(sentences)
        
        # Find common themes (simplified approach)
        word_counts = Counter()
        for sentence in all_sentences:
            words = re.findall(r'\b\w{4,}\b', sentence.lower())  # Words 4+ chars
            word_counts.update(words)
        
        common_themes = [word for word, count in word_counts.most_common(10) if count > 1]
        
        # Look for numerical data
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', ' '.join(all_sentences))
        findings['data_points'] = list(set(numbers))[:10]  # Top 10 unique numbers
        
        # Simplified agreement detection
        if len(source_sentences) > 1:
            findings['agreements'] = [f"Multiple sources mention: {theme}" for theme in common_themes[:3]]
        
        return findings
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences from research text"""
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for key sentences (containing important indicators)
        key_indicators = [
            'research shows', 'study found', 'according to', 'data indicates',
            'results suggest', 'analysis reveals', 'evidence shows', 'reported that',
            'concluded that', 'demonstrated that', 'increased', 'decreased',
            'growth', 'decline', 'significant', 'important', 'critical'
        ]
        
        key_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and 
                any(indicator in sentence.lower() for indicator in key_indicators)):
                key_sentences.append(sentence)
        
        return key_sentences[:5]  # Top 5 key sentences
    
    def _format_key_findings(self, findings: Dict[str, List[str]]) -> str:
        """Format key findings summary"""
        result = "**Key Research Synthesis:**\n\n"
        
        if findings['agreements']:
            result += "**Common Themes:**\n"
            for agreement in findings['agreements']:
                result += f"• {agreement}\n"
            result += "\n"
        
        if findings['data_points']:
            result += "**Key Data Points:**\n"
            for data in findings['data_points'][:5]:
                result += f"• {data}\n"
            result += "\n"
        
        if findings['unique_insights']:
            result += "**Unique Insights:**\n"
            for insight in findings['unique_insights']:
                result += f"• {insight}\n"
            result += "\n"
        
        return result
    
    def _format_research_quality_assessment(self, quality_scores: Dict[str, Dict]) -> str:
        """Format overall research quality assessment"""
        if not quality_scores:
            return ""
        
        result = "**Research Quality Assessment:**\n\n"
        
        # Calculate average quality metrics
        avg_overall = sum(scores['overall'] for scores in quality_scores.values()) / len(quality_scores)
        avg_authority = sum(scores['authority'] for scores in quality_scores.values()) / len(quality_scores)
        avg_recency = sum(scores['recency'] for scores in quality_scores.values()) / len(quality_scores)
        avg_specificity = sum(scores['specificity'] for scores in quality_scores.values()) / len(quality_scores)
        
        result += f"• Overall Research Quality: {avg_overall:.2f}/1.0\n"
        result += f"• Source Authority: {avg_authority:.2f}/1.0\n"
        result += f"• Information Recency: {avg_recency:.2f}/1.0\n"
        result += f"• Data Specificity: {avg_specificity:.2f}/1.0\n"
        result += f"• Sources Consulted: {len(quality_scores)}\n\n"
        
        # Quality interpretation
        if avg_overall >= 0.8:
            quality_level = "Excellent"
        elif avg_overall >= 0.6:
            quality_level = "Good"
        elif avg_overall >= 0.4:
            quality_level = "Moderate"
        else:
            quality_level = "Limited"
        
        result += f"**Research Reliability: {quality_level}**\n"
        
        if avg_authority >= 0.8:
            result += "• High-authority sources with strong credibility\n"
        if avg_recency >= 0.7:
            result += "• Current and up-to-date information\n"
        if avg_specificity >= 0.6:
            result += "• Specific data points and quantitative evidence\n"
        
        return result
    
    def generate_research_queries(self, question: str, current_discussion: List[Dict]) -> List[str]:
        """Auto-generate targeted research queries based on discussion gaps"""
        
        # Analyze discussion for gaps
        discussion_text = "\n".join([msg.get('text', '') for msg in current_discussion])
        
        # Extract claims that need verification
        unsubstantiated_claims = self._find_unsubstantiated_claims(discussion_text)
        
        # Generate specific queries
        queries = []
        
        # Add queries for unsubstantiated claims
        for claim in unsubstantiated_claims[:3]:
            query = self._convert_claim_to_query(claim)
            if query:
                queries.append(query)
        
        # Add queries for missing quantitative data
        if not re.search(r'\d+%', discussion_text):
            queries.append(f"{question} statistics data percentages")
        
        # Add current trends query
        queries.append(f"{question} 2024 2025 recent developments")
        
        return queries[:3]  # Limit to 3 targeted queries
    
    def _find_unsubstantiated_claims(self, discussion_text: str) -> List[str]:
        """Find claims that might need research backing"""
        claims = []
        
        # Look for assertion patterns
        assertion_patterns = [
            r'(?:should|must|will|is|are)\s+[^.]{20,100}',
            r'(?:studies show|research indicates|data suggests)\s+[^.]{20,100}',
            r'(?:according to|based on)\s+[^.]{20,100}'
        ]
        
        for pattern in assertion_patterns:
            matches = re.findall(pattern, discussion_text, re.IGNORECASE)
            claims.extend(matches[:2])  # Limit matches per pattern
        
        return claims
    
    def _convert_claim_to_query(self, claim: str) -> Optional[str]:
        """Convert a claim into a research query"""
        if not claim or len(claim) < 10:
            return None
        
        # Extract key terms
        key_terms = re.findall(r'\b\w{4,}\b', claim.lower())
        if len(key_terms) < 2:
            return None
        
        # Create query from key terms
        query_terms = key_terms[:4]  # Use first 4 meaningful terms
        return " ".join(query_terms)
    
    def prioritize_research_needs(self, expert_positions: List[Dict], question: str) -> List[str]:
        """Identify and prioritize research that could resolve expert conflicts"""
        
        # Extract expert claims
        expert_claims = {}
        for position in expert_positions:
            speaker = position.get('speaker', 'Unknown')
            text = position.get('text', '')
            expert_claims[speaker] = self._extract_key_claims(text)
        
        # Find disagreements
        disagreements = self._find_expert_disagreements(expert_claims)
        
        # Generate research priorities
        priorities = []
        
        for disagreement in disagreements[:3]:
            # Create research query to resolve disagreement
            query = f"{question} {disagreement['topic']} evidence data"
            priorities.append(query)
        
        return priorities
    
    def _extract_key_claims(self, expert_text: str) -> List[str]:
        """Extract key factual claims from expert response"""
        if not expert_text:
            return []
        
        sentences = expert_text.split('.')
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 20 and 
                any(indicator in sentence.lower() for indicator in [
                    'should', 'will', 'is', 'are', 'must', 'can', 'would', 'could'
                ])):
                claims.append(sentence)
        
        return claims[:3]  # Top 3 claims
    
    def _find_expert_disagreements(self, expert_claims: Dict[str, List[str]]) -> List[Dict]:
        """Identify areas where experts disagree"""
        disagreements = []
        
        experts = list(expert_claims.keys())
        
        for i, expert1 in enumerate(experts):
            for expert2 in experts[i+1:]:
                claims1 = expert_claims[expert1]
                claims2 = expert_claims[expert2]
                
                conflicts = self._find_conflicting_claims(claims1, claims2)
                if conflicts:
                    disagreements.append({
                        'experts': [expert1, expert2],
                        'topic': self._extract_conflict_topic(conflicts[0]),
                        'conflicts': conflicts[:1]  # Just the main conflict
                    })
        
        return disagreements
    
    def _find_conflicting_claims(self, claims1: List[str], claims2: List[str]) -> List[str]:
        """Identify potentially conflicting claims (simplified)"""
        conflicts = []
        
        # Simple opposing sentiment detection
        opposing_pairs = [
            ('should', 'should not'), ('will', 'will not'), ('is', 'is not'),
            ('increase', 'decrease'), ('better', 'worse'), ('yes', 'no'),
            ('support', 'oppose'), ('benefit', 'harm'), ('effective', 'ineffective')
        ]
        
        for claim1 in claims1:
            for claim2 in claims2:
                for pos, neg in opposing_pairs:
                    if pos in claim1.lower() and neg in claim2.lower():
                        conflicts.append(f"{claim1} vs {claim2}")
                    elif neg in claim1.lower() and pos in claim2.lower():
                        conflicts.append(f"{claim1} vs {claim2}")
        
        return conflicts
    
    def _extract_conflict_topic(self, conflict: str) -> str:
        """Extract the main topic from a conflict description"""
        # Simple extraction of key terms
        words = re.findall(r'\b\w{4,}\b', conflict.lower())
        # Filter out common words
        stopwords = {'should', 'will', 'would', 'could', 'this', 'that', 'with', 'from', 'they', 'them'}
        topic_words = [word for word in words if word not in stopwords]
        return " ".join(topic_words[:3])
    
    def suggest_research_follow_ups(self, discussion_log: List[Dict], question: str) -> List[str]:
        """Suggest additional research questions based on discussion patterns"""
        
        # Get recent discussion
        latest_messages = discussion_log[-6:] if len(discussion_log) > 6 else discussion_log
        recent_text = "\n".join([msg.get('content', '') for msg in latest_messages])
        
        follow_ups = []
        
        # Look for unverified statistics
        if re.search(r'\d+%', recent_text):
            follow_ups.append(f"{question} statistics verification current data")
        
        # Look for trend mentions
        trend_keywords = ['trend', 'growing', 'increasing', 'declining', 'emerging']
        if any(keyword in recent_text.lower() for keyword in trend_keywords):
            follow_ups.append(f"{question} current trends 2024 2025")
        
        # Look for example mentions
        if 'example' in recent_text.lower() or 'case study' in recent_text.lower():
            follow_ups.append(f"{question} case studies examples evidence")
        
        return follow_ups[:3]
    
    def get_tool_status(self) -> Dict[str, bool]:
        """Get status of all research tools"""
        return {
            name: self.tool_status.get(name, True) 
            for name in self.tools.keys()
        }
    
    def test_tool_connections(self) -> Dict[str, str]:
        """Test all research tool connections"""
        results = {}
        
        for name, tool in self.tools.items():
            try:
                # Simple test query
                test_result = tool.search("test", max_results=1)
                if test_result and len(test_result) > 20:
                    results[name] = "✅ Working"
                    self.tool_status[name] = True
                else:
                    results[name] = "⚠️ Limited response"
                    self.tool_status[name] = False
            except Exception as e:
                results[name] = f"❌ Error: {str(e)[:50]}..."
                self.tool_status[name] = False
        
        return results