"""
GitHub Technology Trends Search Tool
"""
from .base_tool import BaseTool
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class GitHubSearchTool(BaseTool):
    """Search GitHub for technology trends and adoption patterns"""
    
    def __init__(self):
        super().__init__("GitHub", "Search GitHub for technology adoption and development trends")
        self.base_url = "https://api.github.com"
        self.rate_limit_delay = 2.0  # GitHub has rate limits
    
    def search(self, technology: str, max_results: int = 5, **kwargs) -> str:
        """Search GitHub for technology trends and adoption"""
        self.rate_limit()
        
        try:
            # Search repositories
            repos_data = self._search_repositories(technology, max_results)
            
            if not repos_data or not repos_data.get('items'):
                return f"**GitHub Technology Research for: {technology}**\n\nNo relevant repositories found."
            
            result = f"**GitHub Technology Trends for: {technology}**\n\n"
            
            # Repository analysis
            result += self._format_repository_data(repos_data['items'], technology)
            
            # Trend analysis
            result += self._analyze_technology_trends(repos_data, technology)
            
            # Recent activity analysis
            result += self._analyze_recent_activity(repos_data['items'], technology)
            
            return result
            
        except requests.RequestException as e:
            return self.format_error_response(technology, f"Network error accessing GitHub: {str(e)}")
        except Exception as e:
            return self.format_error_response(technology, str(e))
    
    def _search_repositories(self, technology: str, max_results: int) -> Optional[Dict]:
        """Search GitHub repositories for the technology"""
        repos_url = f"{self.base_url}/search/repositories"
        
        # Create comprehensive search query
        search_query = f'{technology} language:python OR language:javascript OR language:typescript OR language:go OR language:rust'
        
        params = {
            'q': search_query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_results
        }
        
        response = requests.get(repos_url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    
    def _format_repository_data(self, repositories: List[Dict], technology: str) -> str:
        """Format repository information"""
        result = f"**Top {len(repositories)} Repositories:**\n"
        
        for i, repo in enumerate(repositories, 1):
            stars = repo.get('stargazers_count', 0)
            forks = repo.get('forks_count', 0)
            language = repo.get('language', 'Unknown')
            updated = repo.get('updated_at', '')[:10]  # YYYY-MM-DD
            
            result += f"**{i}. {repo['name']}** ({stars:,} ⭐, {forks:,} 🍴)\n"
            result += f"   Language: {language} | Updated: {updated}\n"
            
            description = repo.get('description', 'No description')
            if description and len(description) > 100:
                description = description[:100] + "..."
            result += f"   Description: {description}\n"
            result += f"   URL: {repo.get('html_url', 'N/A')}\n\n"
        
        return result
    
    def _analyze_technology_trends(self, repos_data: Dict, technology: str) -> str:
        """Analyze technology adoption trends"""
        total_count = repos_data.get('total_count', 0)
        items = repos_data.get('items', [])
        
        if not items:
            return ""
        
        # Calculate adoption metrics
        total_stars = sum(repo.get('stargazers_count', 0) for repo in items)
        total_forks = sum(repo.get('forks_count', 0) for repo in items)
        avg_stars = total_stars / len(items) if items else 0
        
        # Determine adoption level
        if total_count > 50000:
            adoption_level = "Very High"
        elif total_count > 10000:
            adoption_level = "High"
        elif total_count > 1000:
            adoption_level = "Moderate"
        elif total_count > 100:
            adoption_level = "Emerging"
        else:
            adoption_level = "Niche"
        
        # Language analysis
        languages = {}
        for repo in items:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        result = f"**Technology Adoption Analysis:**\n"
        result += f"• Total repositories: {total_count:,}\n"
        result += f"• Adoption level: {adoption_level}\n"
        result += f"• Average stars (top repos): {avg_stars:,.0f}\n"
        result += f"• Total community engagement: {total_stars:,} stars, {total_forks:,} forks\n"
        
        if languages:
            top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:3]
            result += f"• Popular languages: {', '.join(f'{lang} ({count})' for lang, count in top_languages)}\n"
        
        result += "\n"
        return result
    
    def _analyze_recent_activity(self, repositories: List[Dict], technology: str) -> str:
        """Analyze recent development activity"""
        if not repositories:
            return ""
        
        # Check update recency
        current_date = datetime.now()
        recent_updates = 0
        very_recent_updates = 0
        
        for repo in repositories:
            updated_str = repo.get('updated_at', '')
            if updated_str:
                try:
                    updated_date = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                    days_ago = (current_date - updated_date.replace(tzinfo=None)).days
                    
                    if days_ago <= 30:
                        very_recent_updates += 1
                    if days_ago <= 90:
                        recent_updates += 1
                except:
                    pass
        
        result = f"**Development Activity:**\n"
        result += f"• Recently updated (30 days): {very_recent_updates}/{len(repositories)} repositories\n"
        result += f"• Active projects (90 days): {recent_updates}/{len(repositories)} repositories\n"
        
        # Activity assessment
        if very_recent_updates / len(repositories) > 0.7:
            activity_level = "Very Active"
        elif recent_updates / len(repositories) > 0.5:
            activity_level = "Active"
        elif recent_updates / len(repositories) > 0.3:
            activity_level = "Moderate"
        else:
            activity_level = "Low"
        
        result += f"• Overall activity level: {activity_level}\n"
        result += f"• Community health: {'Strong' if activity_level in ['Very Active', 'Active'] else 'Moderate'} developer engagement\n\n"
        
        return result
    
    def should_use_for_query(self, query: str) -> bool:
        """GitHub is good for technology, framework, and development-related queries"""
        tech_indicators = [
            'technology', 'framework', 'library', 'software', 'programming',
            'development', 'developer', 'code', 'github', 'open source',
            'javascript', 'python', 'react', 'nodejs', 'django', 'flask',
            'vue', 'angular', 'typescript', 'rust', 'go', 'kotlin',
            'adoption', 'popular', 'trending', 'tools', 'stack'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in tech_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from GitHub results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for GitHub-specific patterns
            base_info.update({
                'repo_count': text.count('repositories'),
                'has_stars': '⭐' in text,
                'has_forks': '🍴' in text,
                'has_recent_activity': any(year in text for year in ['2024', '2025']),
                'adoption_mentioned': any(term in text.lower() for term in ['adoption', 'popular', 'trending']),
                'languages_analyzed': 'Popular languages:' in text
            })
        
        return base_info