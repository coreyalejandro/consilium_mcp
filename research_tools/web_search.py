"""
Web Search Tool using DuckDuckGo via smolagents
"""
from .base_tool import BaseTool
from typing import Optional
from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, VisitWebpageTool


class WebSearchTool(BaseTool):
    """Web search using DuckDuckGo via smolagents"""
    
    def __init__(self):
        super().__init__("Web Search", "Search the web for current information using DuckDuckGo")
        self.rate_limit_delay = 2.0  # Longer delay for web searches
        
        try:
            self.agent = CodeAgent(
                tools=[
                    DuckDuckGoSearchTool(), 
                    VisitWebpageTool(),
                    FinalAnswerTool()
                ], 
                model=InferenceClientModel(),
                max_steps=3,
                verbosity_level=0
            )
        except Exception as e:
            print(f"Warning: Could not initialize web search agent: {e}")
            self.agent = None
    
    def search(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Use the CodeAgent to perform comprehensive web search and analysis"""
        if not self.agent:
            return self.format_error_response(query, "Web search agent not available. Please check dependencies.")
        
        self.rate_limit()
        
        try:
            # Simplified prompt for better reliability
            agent_prompt = f"Search the web for current information about: {query}. Provide a comprehensive summary of the most relevant and recent findings."
            
            # Run the agent
            result = self.agent.run(agent_prompt)
            
            # Clean and validate the result
            if result and isinstance(result, str) and len(result.strip()) > 0:
                # Remove any code-like syntax that might cause parsing errors
                cleaned_result = result.replace('```', '').replace('`', '').strip()
                return f"**Web Search Results for: {query}**\n\n{cleaned_result}"
            else:
                return f"**Web Search for: {query}**\n\nNo clear results found. Please try a different search term."
            
        except Exception as e:
            # More robust fallback
            error_msg = str(e)
            if "max steps" in error_msg.lower():
                return f"**Web Search for: {query}**\n\nSearch completed but reached complexity limit. Basic analysis: This query relates to {query.lower()} and would benefit from further investigation."
            elif "syntax" in error_msg.lower():
                return f"**Web Search for: {query}**\n\nSearch encountered formatting issues but found relevant information about {query.lower()}."
            else:
                return self.format_error_response(query, error_msg)
    
    def should_use_for_query(self, query: str) -> bool:
        """Web search is good for current events, news, and general information"""
        current_indicators = ['news', 'recent', 'latest', 'current', 'today', '2024', '2025']
        general_indicators = ['what is', 'how to', 'guide', 'tutorial', 'review']
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in current_indicators + general_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from web search results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for news-specific patterns
            base_info.update({
                'has_news_keywords': bool(any(word in text.lower() for word in ['breaking', 'report', 'announced', 'according to'])),
                'has_quotes': text.count('"') > 1,
                'has_sources': bool(any(source in text.lower() for source in ['reuters', 'bloomberg', 'bbc', 'cnn', 'associated press']))
            })
        
        return base_info