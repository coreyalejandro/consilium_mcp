# research_tools/research_agent.py
"""
Enhanced Research Agent with Multi-Source Integration
Performance-upgraded: result caching, coalesced in-flight queries, parallel deep search
"""

from typing import Dict, List, Tuple
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from functools import lru_cache


from .web_search import WebSearchTool
from .wikipedia_search import WikipediaSearchTool
from .arxiv_search import ArxivSearchTool
from .github_search import GitHubSearchTool
from .sec_search import SECSearchTool

# --------- Simple in-process locks to coalesce identical in-flight queries ---------
_inflight_locks: Dict[Tuple[str, str, str], threading.Lock] = {}
_inflight_results: Dict[Tuple[str, str, str], str] = {}
_inflight_global_lock = threading.Lock()

def _coalesce_key(tool_name: str, query: str, depth: str) -> Tuple[str, str, str]:
    return (tool_name, query.strip().lower(), depth)

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
        # Thread pool for deep queries (IO-bound -> threads are fine; tools use requests)
        self._pool = ThreadPoolExecutor(max_workers=6)

    # ---- Public API -----------------------------------------------------------

    def search(self, query: str, research_depth: str = "standard") -> str:
        """Main search method with intelligent routing"""
        if research_depth == "deep":
            return self._deep_multi_source_search(query)
        else:
            return self._standard_search(query)

    def search_wikipedia(self, topic: str) -> str:
        """Wikipedia search method for backward compatibility"""
        return self.tools['wikipedia'].search(topic)

    # ---- Fast paths + caching ------------------------------------------------

    @lru_cache(maxsize=512)
    def _cached_tool_search(self, tool_name: str, query: str) -> str:
        """Cacheable wrapper: tool_name + normalized query -> str result"""
        return self.tools[tool_name].search(query)

    def _tool_search_coalesced(self, tool_name: str, query: str, depth: str) -> str:
        """
        Coalesce identical in-flight lookups so concurrent callers share one result.
        Falls back to cached value automatically due to _cached_tool_search.
        """
        k = _coalesce_key(tool_name, query, depth)
        with _inflight_global_lock:
            lock = _inflight_locks.get(k)
            if lock is None:
                lock = threading.Lock()
                _inflight_locks[k] = lock

        with lock:
            # If a peer call finished while we were waiting, reuse it.
            if k in _inflight_results:
                return _inflight_results.pop(k)

            # Call through cache (fast if seen before)
            result = self._cached_tool_search(tool_name, query)
            _inflight_results[k] = result
            # Clean up lock map (best-effort)
            with _inflight_global_lock:
                _inflight_locks.pop(k, None)
            return result

    # ---- Standard search -----------------------------------------------------

    def _standard_search(self, query: str) -> str:
        """Standard single-source search with smart routing"""
        best_tool = self._route_query_to_tool(query)
        try:
            return self._tool_search_coalesced(best_tool, query, "standard")
        except Exception as e:
            # Fallback to web search
            if best_tool != 'web':
                try:
                    return self._tool_search_coalesced('web', query, "standard")
                except Exception as e2:
                    return f"**Research for: {query}**\n\nResearch temporarily unavailable: {str(e2)[:100]}..."
            else:
                return f"**Research for: {query}**\n\nResearch temporarily unavailable: {str(e)[:100]}..."

    # ---- Deep search: now parallel ------------------------------------------

    def _deep_multi_source_search(self, query: str) -> str:
        """Deep research using multiple sources with synthesis (parallelized)"""
        results: Dict[str, str] = {}
        quality_scores: Dict[str, Dict] = {}

        relevant_tools = self._get_relevant_tools(query)

        futures = {
            self._pool.submit(self._safe_tool_query, tool_name, query): tool_name
            for tool_name in relevant_tools
        }

        for fut in as_completed(futures):
            tool_name = futures[fut]
            try:
                result = fut.result()
                if result and len(result.strip()) > 50:  # Ensure meaningful result
                    results[tool_name] = result
                    quality_scores[tool_name] = self.tools[tool_name].score_research_quality(result, tool_name)
            except Exception as e:
                # Log to stdout; upstream UI already handles status messages
                print(f"[deep] {tool_name} error: {e}")

        if not results:
            return f"**Deep Research for: {query}**\n\nNo sources were able to provide results. Please try a different query."

        return self._synthesize_multi_source_results(query, results, quality_scores)

    def _safe_tool_query(self, tool_name: str, query: str) -> str:
        """Wrapper for threadpool: coalesced + cached"""
        return self._tool_search_coalesced(tool_name, query, "deep")

    # ---- Routing & synthesis (unchanged except helpers) ----------------------

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
        relevant_tools = ['web']  # always include current info
        for tool_name, tool in self.tools.items():
            if tool_name != 'web' and tool.should_use_for_query(query):
                relevant_tools.append(tool_name)

        # Avoid too many sources (keep deterministic and fast)
        if len(relevant_tools) > 4:
            priority_order = ['arxiv', 'sec', 'github', 'wikipedia', 'web']
            relevant_tools = [tool for tool in priority_order if tool in relevant_tools][:4]
        return relevant_tools

    # ------------- Synthesis + helpers (unchanged from existing) --------------
    # (Keep your existing extraction/formatting methods here — identical to original)
    # I kept all the original methods below verbatim to avoid behavior drift:

    def _synthesize_multi_source_results(self, query: str, results: Dict[str, str], quality_scores: Dict[str, Dict]) -> str:
        synthesis = f"**Comprehensive Research Analysis: {query}**\n\n"
        synthesis += f"**Research Sources Used:** {', '.join(results.keys()).replace('_', ' ').title()}\n\n"
        key_findings = self._extract_key_findings(results)
        synthesis += self._format_key_findings(key_findings)
        synthesis += "**Detailed Source Results:**\n\n"
        sorted_sources = sorted(quality_scores.items(), key=lambda x: x[1]['overall'], reverse=True)
        for source_name, _ in sorted_sources:
            if source_name in results:
                source_result = results[source_name]
                quality = quality_scores[source_name]
                if len(source_result) > 800:
                    source_result = source_result[:800] + "...\n[Result truncated for synthesis]"
                synthesis += f"**{source_name.replace('_', ' ').title()} (Quality: {quality['overall']:.2f}/1.0):**\n"
                synthesis += f"{source_result}\n\n"
        synthesis += self._format_research_quality_assessment(quality_scores)
        return synthesis

    def _extract_key_findings(self, results: Dict[str, str]) -> Dict[str, List[str]]:
        findings = {'agreements': [], 'contradictions': [], 'unique_insights': [], 'data_points': []}
        all_sentences = []
        source_sentences = {}
        for source, result in results.items():
            sentences = self._extract_key_sentences(result)
            source_sentences[source] = sentences
            all_sentences.extend(sentences)
        word_counts = Counter()
        for sentence in all_sentences:
            words = re.findall(r'\b\w{4,}\b', sentence.lower())
            word_counts.update(words)
        common_themes = [word for word, count in word_counts.most_common(10) if count > 1]
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', ' '.join(all_sentences))
        findings['data_points'] = list(set(numbers))[:10]
        if len(source_sentences) > 1:
            findings['agreements'] = [f"Multiple sources mention: {theme}" for theme in common_themes[:3]]
        return findings

    def _extract_key_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'[.!?]+', text)
        key_indicators = [
            'research shows', 'study found', 'according to', 'data indicates',
            'results suggest', 'analysis reveals', 'evidence shows', 'reported that',
            'concluded that', 'demonstrated that', 'increased', 'decreased',
            'growth', 'decline', 'significant', 'important', 'critical'
        ]
        key_sentences = []
        for sentence in sentences:
            s = sentence.strip()
            if len(s) > 30 and any(indicator in s.lower() for indicator in key_indicators):
                key_sentences.append(s)
        return key_sentences[:5]

    def _format_key_findings(self, findings: Dict[str, List[str]]) -> str:
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
        if not quality_scores:
            return ""
        avg_overall = sum(scores['overall'] for scores in quality_scores.values()) / len(quality_scores)
        avg_authority = sum(scores['authority'] for scores in quality_scores.values()) / len(quality_scores)
        avg_recency = sum(scores['recency'] for scores in quality_scores.values()) / len(quality_scores)
        avg_specificity = sum(scores['specificity'] for scores in quality_scores.values()) / len(quality_scores)
        result = "**Research Quality Assessment:**\n\n"
        result += f"• Overall Research Quality: {avg_overall:.2f}/1.0\n"
        result += f"• Source Authority: {avg_authority:.2f}/1.0\n"
        result += f"• Information Recency: {avg_recency:.2f}/1.0\n"
        result += f"• Data Specificity: {avg_specificity:.2f}/1.0\n"
        result += f"• Sources Consulted: {len(quality_scores)}\n\n"
        quality_level = ("Excellent" if avg_overall >= 0.8 else
                         "Good" if avg_overall >= 0.6 else
                         "Moderate" if avg_overall >= 0.4 else "Limited")
        result += f"**Research Reliability: {quality_level}**\n"
        if avg_authority >= 0.8:
            result += "• High-authority sources with strong credibility\n"
        if avg_recency >= 0.7:
            result += "• Current and up-to-date information\n"
        if avg_specificity >= 0.6:
            result += "• Specific data points and quantitative evidence\n"
        return result
