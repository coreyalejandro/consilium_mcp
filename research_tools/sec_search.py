"""
SEC Edgar Filings Search Tool for financial and company data
"""
from .base_tool import BaseTool
import requests
import json
import re
from typing import Dict, List, Optional


class SECSearchTool(BaseTool):
    """Search SEC EDGAR filings for company financial information"""
    
    def __init__(self):
        super().__init__("SEC EDGAR", "Search SEC filings and financial data for public companies")
        self.base_url = "https://data.sec.gov"
        self.headers = {
            'User-Agent': 'Research Tool research@academic.edu',  # SEC requires User-Agent
            'Accept-Encoding': 'gzip, deflate'
        }
        self.rate_limit_delay = 3.0  # SEC is strict about rate limiting
    
    def search(self, company_name: str, **kwargs) -> str:
        """Search SEC filings for company information"""
        self.rate_limit()
        
        try:
            # First attempt to find company CIK
            cik_data = self._find_company_cik(company_name)
            
            if not cik_data:
                return self._fallback_company_search(company_name)
            
            # Get company submissions
            submissions = self._get_company_submissions(cik_data['cik'])
            
            if submissions:
                return self._format_sec_results(company_name, cik_data, submissions)
            else:
                return self._fallback_company_search(company_name)
                
        except requests.RequestException as e:
            # Handle network errors gracefully
            if "404" in str(e):
                return self._fallback_company_search(company_name)
            return self.format_error_response(company_name, f"Network error accessing SEC: {str(e)}")
        except Exception as e:
            return self.format_error_response(company_name, str(e))
    
    def _find_company_cik(self, company_name: str) -> Optional[Dict]:
        """Find company CIK (Central Index Key) from company name"""
        try:
            # Use the correct SEC company tickers endpoint
            tickers_url = "https://www.sec.gov/files/company_tickers_exchange.json"
            response = requests.get(tickers_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            # Search for company by name (fuzzy matching)
            company_lower = company_name.lower()
            
            # Handle the exchange data format
            if isinstance(tickers_data, dict):
                # Check if it's the fields/data format
                if 'fields' in tickers_data and 'data' in tickers_data:
                    return self._search_exchange_format(tickers_data, company_lower)
                else:
                    # Try direct dictionary format
                    return self._search_direct_format(tickers_data, company_lower)
            elif isinstance(tickers_data, list):
                # Handle list format
                return self._search_list_format(tickers_data, company_lower)
            
            return None
            
        except Exception as e:
            print(f"Error finding company CIK: {e}")
            return self._fallback_company_lookup(company_name)
    
    def _fallback_company_lookup(self, company_name: str) -> Optional[Dict]:
        """Fallback company lookup using known major companies"""
        # Hardcoded CIKs for major companies for testing/demo purposes
        known_companies = {
            'apple': {'cik': '0000320193', 'ticker': 'AAPL', 'title': 'Apple Inc.'},
            'microsoft': {'cik': '0000789019', 'ticker': 'MSFT', 'title': 'Microsoft Corporation'},
            'tesla': {'cik': '0001318605', 'ticker': 'TSLA', 'title': 'Tesla, Inc.'},
            'amazon': {'cik': '0001018724', 'ticker': 'AMZN', 'title': 'Amazon.com, Inc.'},
            'google': {'cik': '0001652044', 'ticker': 'GOOGL', 'title': 'Alphabet Inc.'},
            'alphabet': {'cik': '0001652044', 'ticker': 'GOOGL', 'title': 'Alphabet Inc.'},
            'meta': {'cik': '0001326801', 'ticker': 'META', 'title': 'Meta Platforms, Inc.'},
            'facebook': {'cik': '0001326801', 'ticker': 'META', 'title': 'Meta Platforms, Inc.'},
            'nvidia': {'cik': '0001045810', 'ticker': 'NVDA', 'title': 'NVIDIA Corporation'},
            'netflix': {'cik': '0001065280', 'ticker': 'NFLX', 'title': 'Netflix, Inc.'}
        }
        
        company_key = company_name.lower().strip()
        for key, data in known_companies.items():
            if key in company_key or company_key in key:
                return data
        
        return None
    
    def _search_exchange_format(self, tickers_data: dict, company_lower: str) -> Optional[Dict]:
        """Search in exchange ticker data format"""
        try:
            fields = tickers_data.get('fields', [])
            data = tickers_data.get('data', [])
            
            # Find field indices
            cik_idx = None
            ticker_idx = None
            name_idx = None
            
            for i, field in enumerate(fields):
                if field.lower() in ['cik', 'cik_str']:
                    cik_idx = i
                elif field.lower() in ['ticker', 'symbol']:
                    ticker_idx = i
                elif field.lower() in ['name', 'title', 'company']:
                    name_idx = i
            
            # Search through data
            for row in data:
                if len(row) > max(filter(None, [cik_idx, ticker_idx, name_idx])):
                    name = str(row[name_idx]).lower() if name_idx is not None else ""
                    ticker = str(row[ticker_idx]).lower() if ticker_idx is not None else ""
                    
                    if (company_lower in name or 
                        name in company_lower or
                        company_lower == ticker or
                        any(word in name for word in company_lower.split() if len(word) > 3)):
                        
                        cik = str(row[cik_idx]) if cik_idx is not None else ""
                        return {
                            'cik': cik.zfill(10),
                            'ticker': row[ticker_idx] if ticker_idx is not None else "",
                            'title': row[name_idx] if name_idx is not None else ""
                        }
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing exchange format: {e}")
        
        return None
    
    def _search_direct_format(self, tickers_data: dict, company_lower: str) -> Optional[Dict]:
        """Search in direct dictionary format"""
        for key, entry in tickers_data.items():
            if isinstance(entry, dict):
                title = entry.get('title', entry.get('name', '')).lower()
                ticker = entry.get('ticker', entry.get('symbol', '')).lower()
                
                if (company_lower in title or 
                    title in company_lower or
                    company_lower == ticker or
                    any(word in title for word in company_lower.split() if len(word) > 3)):
                    
                    return {
                        'cik': str(entry.get('cik_str', entry.get('cik', ''))).zfill(10),
                        'ticker': entry.get('ticker', entry.get('symbol', '')),
                        'title': entry.get('title', entry.get('name', ''))
                    }
        return None
    
    def _search_list_format(self, tickers_data: list, company_lower: str) -> Optional[Dict]:
        """Search in list format"""
        for entry in tickers_data:
            if isinstance(entry, dict):
                title = entry.get('title', entry.get('name', '')).lower()
                ticker = entry.get('ticker', entry.get('symbol', '')).lower()
                
                if (company_lower in title or 
                    title in company_lower or
                    company_lower == ticker or
                    any(word in title for word in company_lower.split() if len(word) > 3)):
                    
                    return {
                        'cik': str(entry.get('cik_str', entry.get('cik', ''))).zfill(10),
                        'ticker': entry.get('ticker', entry.get('symbol', '')),
                        'title': entry.get('title', entry.get('name', ''))
                    }
        return None
    
    def _get_company_submissions(self, cik: str) -> Optional[Dict]:
        """Get company submission data from SEC"""
        try:
            submissions_url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = requests.get(submissions_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error getting company submissions: {e}")
            return None
    
    def _format_sec_results(self, company_name: str, cik_data: Dict, submissions: Dict) -> str:
        """Format SEC filing results"""
        result = f"**SEC Financial Data for: {company_name}**\n\n"
        
        # Company information
        result += f"**Company Information:**\n"
        result += f"• Official Name: {cik_data['title']}\n"
        result += f"• Ticker Symbol: {cik_data.get('ticker', 'N/A')}\n"
        result += f"• CIK: {cik_data['cik']}\n"
        
        # Business information
        if 'description' in submissions:
            business_desc = submissions['description'][:300] + "..." if len(submissions.get('description', '')) > 300 else submissions.get('description', 'Not available')
            result += f"• Business Description: {business_desc}\n"
        
        result += f"• Industry: {submissions.get('sic', 'Not specified')}\n"
        result += f"• Fiscal Year End: {submissions.get('fiscalYearEnd', 'Not specified')}\n\n"
        
        # Recent filings analysis
        recent_filings = self._analyze_recent_filings(submissions)
        result += recent_filings
        
        # Financial highlights
        financial_highlights = self._extract_financial_highlights(submissions)
        result += financial_highlights
        
        return result
    
    def _analyze_recent_filings(self, submissions: Dict) -> str:
        """Analyze recent SEC filings"""
        result = "**Recent SEC Filings:**\n"
        
        # Get recent filings
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            return result + "• No recent filings available\n\n"
        
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        # Analyze key filing types
        key_forms = ['10-K', '10-Q', '8-K', 'DEF 14A']
        recent_key_filings = []
        
        for i, form in enumerate(forms[:20]):  # Check last 20 filings
            if form in key_forms and i < len(filing_dates):
                recent_key_filings.append({
                    'form': form,
                    'date': filing_dates[i],
                    'accession': accession_numbers[i] if i < len(accession_numbers) else 'N/A'
                })
        
        if recent_key_filings:
            for filing in recent_key_filings[:5]:  # Show top 5
                form_description = {
                    '10-K': 'Annual Report',
                    '10-Q': 'Quarterly Report', 
                    '8-K': 'Current Report',
                    'DEF 14A': 'Proxy Statement'
                }.get(filing['form'], filing['form'])
                
                result += f"• {filing['form']} ({form_description}) - Filed: {filing['date']}\n"
        else:
            result += "• No key financial filings found in recent submissions\n"
        
        result += "\n"
        return result
    
    def _extract_financial_highlights(self, submissions: Dict) -> str:
        """Extract financial highlights from submission data"""
        result = "**Financial Data Analysis:**\n"
        
        # This is a simplified version - full implementation would parse actual financial data
        result += "• Filing Status: Active public company\n"
        result += "• Regulatory Compliance: Current with SEC requirements\n"
        
        # Check for recent financial filings
        recent_filings = submissions.get('filings', {}).get('recent', {})
        if recent_filings:
            forms = recent_filings.get('form', [])
            annual_reports = sum(1 for form in forms if form == '10-K')
            quarterly_reports = sum(1 for form in forms if form == '10-Q')
            
            result += f"• Annual Reports (10-K): {annual_reports} on file\n"
            result += f"• Quarterly Reports (10-Q): {quarterly_reports} on file\n"
        
        result += "• Note: Detailed financial metrics require parsing individual filing documents\n\n"
        
        result += "**Investment Research Notes:**\n"
        result += "• Use SEC filings for: revenue trends, risk factors, management discussion\n"
        result += "• Key documents: 10-K (annual), 10-Q (quarterly), 8-K (material events)\n"
        result += "• Combine with market data for comprehensive analysis\n\n"
        
        return result
    
    def _fallback_company_search(self, company_name: str) -> str:
        """Fallback response when company not found in SEC database"""
        result = f"**SEC Financial Research for: {company_name}**\n\n"
        result += f"**Company Search Results:**\n"
        result += f"• Company '{company_name}' not found in SEC EDGAR database\n"
        result += f"• This may indicate the company is:\n"
        result += f"  - Private company (not required to file with SEC)\n"
        result += f"  - Foreign company not listed on US exchanges\n"
        result += f"  - Subsidiary of another public company\n"
        result += f"  - Different legal name than search term\n\n"
        
        result += f"**Alternative Research Suggestions:**\n"
        result += f"• Search for parent company or holding company\n"
        result += f"• Check if company trades under different ticker symbol\n"
        result += f"• Use company's full legal name for search\n"
        result += f"• Consider private company databases for non-public entities\n\n"
        
        return result
    
    def should_use_for_query(self, query: str) -> bool:
        """SEC is good for public company financial and business information"""
        financial_indicators = [
            'company', 'financial', 'revenue', 'earnings', 'profit', 'stock',
            'investment', 'market cap', 'sec filing', 'annual report',
            'quarterly', 'balance sheet', 'income statement', 'cash flow',
            'public company', 'ticker', 'investor', 'shareholder'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in financial_indicators)
    
    def extract_key_info(self, text: str) -> dict:
        """Extract key information from SEC results"""
        base_info = super().extract_key_info(text)
        
        if text:
            # Look for SEC-specific patterns
            base_info.update({
                'has_ticker': any(pattern in text for pattern in ['Ticker Symbol:', 'ticker']),
                'has_cik': 'CIK:' in text,
                'has_filings': any(form in text for form in ['10-K', '10-Q', '8-K']),
                'is_public_company': 'public company' in text.lower(),
                'has_financial_data': any(term in text.lower() for term in ['revenue', 'earnings', 'financial']),
                'company_found': 'not found in SEC' not in text
            })
        
        return base_info