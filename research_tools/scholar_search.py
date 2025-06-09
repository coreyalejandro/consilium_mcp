"""
Google Scholar Search Tool for academic research
"""
from .base_tool import BaseTool
from typing import List, Dict, Optional

try:
    from scholarly import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False


class GoogleScholarTool(BaseTool):
    """Search Google Scholar for academic research papers"""
    
    def __init__(self):
        super().__init__("Google Scholar", "Search Google Scholar for academic research papers and citations")
        self.available = SCHOLARLY_AVAILABLE
        self.rate_limit_delay = 3.0  # Be very respectful to Google Scholar
    
    def search(self, query: str, max_results: int = 4, **kwargs) -> str:
        """Search Google Scholar for research papers"""
        if not self.available:
            return self._unavailable_response(query)
        
        self.rate_limit()
        
        try:
            # Search for publications with timeout handling
            search_query = scholarly.search_pubs(query)
            
            papers = []
            for i, paper in enumerate(search_query):
                if i >= max_results:
                    break
                # Try to get additional info if available
                try:
                    # Some papers might need to be filled for complete info
                    if hasattr(paper, 'fill') and callable(paper.fill):
                        paper = paper.fill()
                except:
                    # If fill fails, use paper as-is
                    pass
                papers.append(paper)
            
            if papers:
                result = f"**Google Scholar Research for: {query}**\n\n"
                result += self._format_scholar_results(papers)
                result += self._analyze_research_quality(papers)
                return result
            else:
                return f"**Google Scholar Research for: {query}**\n\nNo relevant academic papers found."
                
        except Exception as e:
            error_msg = str(e)
            if "blocked" in error_msg.lower() or "captcha" in error_msg.lower():
                return f"**Google Scholar Research for: {query}**\n\nGoogle Scholar is temporarily blocking automated requests. This is normal behavior. Academic research is available through other sources like arXiv."
            elif "timeout" in error_msg.lower():
                return f"**Google Scholar Research for: {query}**\n\nRequest timeout - Google Scholar may be experiencing high load. Academic research available but slower than expected."
            else:
                return self.format_error_response(query, str(e))
    
    def _unavailable_response(self, query: str) -> str:
        """Response when scholarly library is not available"""
        result = f"**Google Scholar Research for: {query}**\n\n"
        result += "**Library Not Available**\n"
        result += "Google Scholar integration requires the 'scholarly' library.\n\n"
        result += "**Installation Instructions:**\n"
        result += "```bash\n"
        result += "pip install scholarly\n"
        result += "```\n\n"
        result += "**Alternative Academic Sources:**\n"
        result += "• arXiv (for preprints and technical papers)\n"
        result += "• PubMed (for medical and life sciences)\n"
        result += "• IEEE Xplore (for engineering and computer science)\n"
        result += "• JSTOR (for humanities and social sciences)\n\n"
        result += "**Research Recommendation:**\n"
        result += f"For the query '{query}', consider searching:\n"
        result += "• Recent academic publications\n"
        result += "• Peer-reviewed research articles\n"
        result += "• Citation networks and impact metrics\n\n"
        
        return result
    
    def _format_scholar_results(self, papers: List[Dict]) -> str:
        """Format Google Scholar search results"""
        result = ""
        
        for i, paper in enumerate(papers, 1):
            # Extract paper information safely with better handling
            title = paper.get('title', paper.get('bib', {}).get('title', 'Unknown Title'))
            
            # Handle authors more robustly
            authors = self._format_authors(paper.get('author', paper.get('bib', {}).get('author', [])))
            
            # Get year from multiple possible locations
            year = (paper.get('year') or 
                   paper.get('bib', {}).get('pub_year') or 
                   paper.get('bib', {}).get('year') or 
                   'Unknown Year')
            
            # Get venue from multiple possible locations
            venue = (paper.get('venue') or 
                    paper.get('bib', {}).get('venue') or 
                    paper.get('bib', {}).get('journal') or 
                    paper.get('bib', {}).get('booktitle') or 
                    'Unknown Venue')
            
            citations = paper.get('num_citations', paper.get('citedby', 0))
            
            result += f"**Paper {i}: {title}**\n"
            result += f"Authors: {authors}\n"
            result += f"Year: {year} | Venue: {venue}\n"
            result += f"Citations: {citations:,}\n"
            
            # Add abstract if available
            abstract = (paper.get('abstract') or 
                       paper.get('bib', {}).get('abstract') or 
                       paper.get('summary'))
            
            if abstract and len(str(abstract).strip()) > 10:
                abstract_text = str(abstract)
                if len(abstract_text) > 300:
                    abstract_text = abstract_text[:300] + "..."
                result += f"Abstract: {abstract_text}\n"
            
            # Add URL if available
            url = (paper.get('url') or 
                  paper.get('pub_url') or 
                  paper.get('eprint_url'))
            
            if url:
                result += f"URL: {url}\n"
            
            result += "\n"
        
        return result
    
    def _format_authors(self, authors) -> str:
        """Format author list safely with improved handling"""
        if not authors:
            return "Unknown Authors"
        
        if isinstance(authors, str):
            return authors
        elif isinstance(authors, list):
            # Handle list of author dictionaries or strings
            author_names = []
            for author in authors[:5]:  # Limit to first 5 authors
                if isinstance(author, dict):
                    # Try different possible name fields
                    name = (author.get('name') or 
                           author.get('full_name') or 
                           author.get('firstname', '') + ' ' + author.get('lastname', '') or
                           str(author))
                    name = name.strip()
                else:
                    name = str(author).strip()
                
                if name and name != 'Unknown Authors':
                    author_names.append(name)
            
            if not author_names:
                return "Unknown Authors"
            
            if len(authors) > 5:
                author_names.append("et al.")
            
            return ", ".join(author_names)
        else:
            return str(authors) if authors else "Unknown Authors"
    
    def _analyze_research_quality(self, papers: List[Dict]) -> str:
        """Analyze the quality and impact of research results"""
        if not papers:
            return ""
        
        # Calculate citation metrics
        citations = [paper.get('num_citations', 0) for paper in papers]
        total_citations = sum(citations)
        avg_citations = total_citations / len(papers) if papers else 0
        high_impact_papers = sum(1 for c in citations if c > 100)
        
        # Analyze publication years
        years = [paper.get('year') for paper in papers if paper.get('year')]
        recent_papers = sum(1 for year in years if isinstance(year, (int, str)) and str(year) in ['2023', '2024', '2025'])
        
        # Analyze venues
        venues = [paper.get('venue', '') for paper in papers]
        unique_venues = len(set(v for v in venues if v and v != 'Unknown Venue'))
        
        result = f"**Research Quality Analysis:**\n"
        result += f"• Papers analyzed: {len(papers)}\n"
        result += f"• Total citations: {total_citations:,}\n"
        result += f"• Average citations per paper: {avg_citations:.1f}\n"
        result += f"• High-impact papers (>100 citations): {high_impact_papers}\n"
        result += f"• Recent publications (2023-2025): {recent_papers}\n"
        result += f"• Venue diversity: {unique_venues} different publication venues\n"
        
        # Research quality assessment
        if avg_citations > 50:
            quality_level = "High Impact"
        elif avg_citations > 20:
            quality_level = "Moderate Impact"
        elif avg_citations > 5:
            quality_level = "Emerging Research"
        else:
            quality_level = "Early Stage"
        
        result += f"• Research maturity: {quality_level}\n"
        
        # Authority assessment
        if high_impact_papers > 0 and recent_papers > 0:
            authority = "High - Established field with recent developments"
        elif high_impact_papers > 0:
            authority = "Moderate - Established field, may need recent updates"
        elif recent_papers > 0:
            authority = "Emerging - New research area with growing interest"
        else:
            authority = "Limited - Sparse academic coverage"
        
        result += f"• Academic authority: {authority}\n\n"
        
        return result
    
    def should_use_for_query(self, query: str) -> bool:
        """Google Scholar is good for academic research, citations, and scholarly articles"""
        academic_indicators = [
            'research', 'study', 'academic', 'paper', 'journal', 'peer-reviewed',
            'citation', 'scholar', 'university', 'professor', 'phd', 'thesis',
            'methodology', 'experiment', 'analysis', 'theory', 'empirical',
            'literature review', 'meta-analysis', 'systematic review',
            'conference', 'publication', 'scholarly'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in academic_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from Scholar results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for Scholar-specific patterns
            base_info.update({
                'has_citations': 'Citations:' in text,
                'has_abstracts': 'Abstract:' in text,
                'has_venues': 'Venue:' in text,
                'has_recent_papers': any(year in text for year in ['2023', '2024', '2025']),
                'has_high_impact': any(citation in text for citation in ['100', '200', '500', '1000']),
                'is_available': 'Library Not Available' not in text,
                'paper_count': text.count('**Paper')
            })
        
        return base_info