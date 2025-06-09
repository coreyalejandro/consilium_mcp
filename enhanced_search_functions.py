"""
Enhanced Search Functions for Native Function Calling
This file defines all the function calling schemas for the enhanced research system
"""

ENHANCED_SEARCH_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information and real-time data using DuckDuckGo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find current information relevant to the expert analysis"
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["standard", "deep"],
                        "description": "Search depth - 'standard' for single source, 'deep' for multi-source synthesis",
                        "default": "standard"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for comprehensive background information and authoritative encyclopedic data",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to research on Wikipedia for comprehensive background information"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_academic",
            "description": "Search academic papers and research on arXiv and Google Scholar for scientific evidence",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Academic research query to find peer-reviewed papers and scientific studies"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["arxiv", "scholar", "both"],
                        "description": "Academic source to search - arXiv for preprints, Scholar for citations, both for comprehensive",
                        "default": "both"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_technology_trends",
            "description": "Search GitHub for technology adoption, development trends, and open source activity",
            "parameters": {
                "type": "object",
                "properties": {
                    "technology": {
                        "type": "string",
                        "description": "Technology, framework, or programming language to research for adoption trends"
                    }
                },
                "required": ["technology"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_data", 
            "description": "Search SEC EDGAR filings and financial data for public companies",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "Company name or ticker symbol to research financial data and SEC filings"
                    }
                },
                "required": ["company"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_source_research",
            "description": "Perform comprehensive multi-source research synthesis across all available sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research query for comprehensive multi-source analysis"
                    },
                    "priority_sources": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["web", "wikipedia", "arxiv", "scholar", "github", "sec"]
                        },
                        "description": "Priority list of sources to focus on for this research",
                        "default": []
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def get_function_definitions():
    """Get the complete function definitions for API calls"""
    return ENHANCED_SEARCH_FUNCTIONS

def get_function_names():
    """Get list of all available function names"""
    return [func["function"]["name"] for func in ENHANCED_SEARCH_FUNCTIONS]

# Function routing map for backward compatibility
FUNCTION_ROUTING = {
    "search_web": "web_search",
    "search_wikipedia": "wikipedia_search", 
    "search_academic": "academic_search",
    "search_technology_trends": "github_search",
    "search_financial_data": "sec_search",
    "multi_source_research": "multi_source_search"
}