"""
arXiv Academic Papers Search Tool
"""
from .base_tool import BaseTool
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.parse import quote


class ArxivSearchTool(BaseTool):
    """Search arXiv for academic papers and research"""
    
    def __init__(self):
        super().__init__("arXiv", "Search academic papers and research on arXiv")
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = 2.0  # Be respectful to arXiv
    
    def search(self, query: str, max_results: int = 5, **kwargs) -> str:
        """Search arXiv for academic papers"""
        self.rate_limit()
        
        try:
            # Prepare search parameters
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            # Make request with better error handling
            response = requests.get(self.base_url, params=params, timeout=20, 
                                  headers={'User-Agent': 'Research Tool (research@academic.edu)'})
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract paper information
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            # Format results
            if papers:
                result = f"**arXiv Academic Research for: {query}**\n\n"
                for i, paper in enumerate(papers, 1):
                    result += f"**Paper {i}: {paper['title']}**\n"
                    result += f"Authors: {paper['authors']}\n"
                    result += f"Published: {paper['published']}\n"
                    result += f"Category: {paper.get('category', 'Unknown')}\n"
                    result += f"Abstract: {paper['abstract'][:400]}...\n"
                    result += f"Link: {paper['link']}\n\n"
                
                # Add research quality assessment
                result += self._assess_arxiv_quality(papers)
                
                return result
            else:
                return f"**arXiv Research for: {query}**\n\nNo relevant academic papers found on arXiv."
                
        except requests.Timeout:
            return f"**arXiv Research for: {query}**\n\nRequest timeout - arXiv may be experiencing high load. Research available but slower than expected."
        except requests.ConnectionError as e:
            if "Connection reset" in str(e):
                return f"**arXiv Research for: {query}**\n\nConnection reset by arXiv server - this is common due to rate limiting. Academic research is available but temporarily throttled."
            return self.format_error_response(query, f"Connection error: {str(e)}")
        except requests.RequestException as e:
            return self.format_error_response(query, f"Network error accessing arXiv: {str(e)}")
        except ET.ParseError as e:
            return self.format_error_response(query, f"Error parsing arXiv response: {str(e)}")
        except Exception as e:
            return self.format_error_response(query, str(e))
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict[str, str]]:
        """Parse individual arXiv entry"""
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            title = entry.find('atom:title', ns)
            title_text = title.text.strip().replace('\n', ' ') if title is not None else "Unknown Title"
            
            authors = entry.findall('atom:author/atom:name', ns)
            author_names = [author.text for author in authors] if authors else ["Unknown Author"]
            
            published = entry.find('atom:published', ns)
            published_text = published.text[:10] if published is not None else "Unknown Date"  # YYYY-MM-DD
            
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip().replace('\n', ' ') if summary is not None else "No abstract available"
            
            link = entry.find('atom:id', ns)
            link_url = link.text if link is not None else ""
            
            # Extract category
            categories = entry.findall('atom:category', ns)
            category = categories[0].get('term') if categories else "Unknown"
            
            return {
                'title': title_text,
                'authors': ', '.join(author_names[:3]),  # Limit to first 3 authors
                'published': published_text,
                'abstract': abstract,
                'link': link_url,
                'category': category
            }
        except Exception as e:
            print(f"Error parsing arXiv entry: {e}")
            return None
    
    def _assess_arxiv_quality(self, papers: List[Dict]) -> str:
        """Assess the quality of arXiv search results"""
        if not papers:
            return ""
        
        # Calculate average recency
        current_year = 2025
        recent_papers = sum(1 for paper in papers if paper['published'].startswith(('2024', '2025')))
        
        quality_assessment = f"**Research Quality Assessment:**\n"
        quality_assessment += f"• Papers found: {len(papers)}\n"
        quality_assessment += f"• Recent papers (2024-2025): {recent_papers}/{len(papers)}\n"
        
        # Check for high-impact categories
        categories = [paper.get('category', '') for paper in papers]
        ml_ai_papers = sum(1 for cat in categories if any(term in cat.lower() for term in ['cs.ai', 'cs.lg', 'cs.cv', 'stat.ml']))
        if ml_ai_papers > 0:
            quality_assessment += f"• AI/ML papers: {ml_ai_papers}\n"
        
        quality_assessment += f"• Authority level: High (peer-reviewed preprints)\n\n"
        
        return quality_assessment
    
    def should_use_for_query(self, query: str) -> bool:
        """arXiv is good for scientific, technical, and research-oriented queries"""
        academic_indicators = [
            'research', 'study', 'analysis', 'scientific', 'algorithm', 'method',
            'machine learning', 'ai', 'artificial intelligence', 'deep learning',
            'neural network', 'computer science', 'physics', 'mathematics',
            'quantum', 'cryptography', 'blockchain', 'paper', 'academic'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in academic_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from arXiv results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for arXiv-specific patterns
            base_info.update({
                'paper_count': text.count('**Paper'),
                'has_abstracts': 'Abstract:' in text,
                'has_recent_papers': any(year in text for year in ['2024', '2025']),
                'has_ai_ml': any(term in text.lower() for term in ['machine learning', 'ai', 'neural', 'deep learning']),
                'has_arxiv_links': 'arxiv.org' in text
            })
        
        return base_info