"""
Wikipedia Search Tool for comprehensive background information
"""
from .base_tool import BaseTool
from typing import Optional


class WikipediaSearchTool(BaseTool):
    """Search Wikipedia for comprehensive background information"""
    
    def __init__(self):
        super().__init__("Wikipedia", "Search Wikipedia for comprehensive background information and authoritative data")
        self.rate_limit_delay = 1.0
    
    def search(self, query: str, max_results: int = 3, **kwargs) -> str:
        """Search Wikipedia for comprehensive information"""
        self.rate_limit()
        
        try:
            import wikipedia
            
            # Search for the topic
            search_results = wikipedia.search(query, results=max_results)
            if not search_results:
                return f"**Wikipedia Research for: {query}**\n\nNo Wikipedia articles found for: {query}"
            
            result = f"**Wikipedia Research for: {query}**\n\n"
            
            for i, search_term in enumerate(search_results[:max_results]):
                try:
                    # Get the page
                    page = wikipedia.page(search_term)
                    summary = page.summary[:800] + "..." if len(page.summary) > 800 else page.summary
                    
                    result += f"**Article {i+1}: {page.title}**\n"
                    result += f"{summary}\n"
                    result += f"Source: {page.url}\n\n"
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = page.summary[:600] + "..." if len(page.summary) > 600 else page.summary
                        result += f"**Article {i+1}: {page.title}**\n"
                        result += f"{summary}\n"
                        result += f"Source: {page.url}\n\n"
                    except:
                        result += f"**Article {i+1}:** Multiple options found for '{search_term}'\n\n"
                        
                except wikipedia.exceptions.PageError:
                    result += f"**Article {i+1}:** Page not found for '{search_term}'\n\n"
                    
                except Exception as e:
                    result += f"**Article {i+1}:** Error accessing '{search_term}': {str(e)[:50]}...\n\n"
            
            return result
            
        except ImportError:
            return f"**Wikipedia Research for: {query}**\n\nWikipedia library not available. Please install with: pip install wikipedia\n\n"
        except Exception as e:
            return self.format_error_response(query, str(e))
    
    def should_use_for_query(self, query: str) -> bool:
        """Wikipedia is good for factual, historical, and encyclopedic information"""
        encyclopedic_indicators = [
            'what is', 'who is', 'history of', 'definition', 'background', 
            'overview', 'explain', 'about', 'biography', 'concept'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in encyclopedic_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from Wikipedia results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for Wikipedia-specific patterns
            base_info.update({
                'has_categories': 'Category:' in text,
                'has_references': any(ref in text for ref in ['Retrieved', 'Archived', 'ISBN']),
                'is_biographical': any(bio in text.lower() for bio in ['born', 'died', 'biography', 'life']),
                'is_historical': any(hist in text.lower() for hist in ['century', 'founded', 'established', 'ancient']),
                'article_count': text.count('**Article')
            })
        
        return base_info