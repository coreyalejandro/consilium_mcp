# Research Tools Package
from .base_tool import BaseTool
from .web_search import WebSearchTool
from .wikipedia_search import WikipediaSearchTool
from .arxiv_search import ArxivSearchTool
from .github_search import GitHubSearchTool
from .sec_search import SECSearchTool
from .research_agent import EnhancedResearchAgent
from .dataset_logger import log_training_example
from .dspy_module import DSPySynthesisProgram
from .report_templates import get_report_template

__all__ = [
    'BaseTool',
    'WebSearchTool', 
    'WikipediaSearchTool',
    'ArxivSearchTool',
    'GitHubSearchTool', 
    'SECSearchTool',
    'EnhancedResearchAgent'
]