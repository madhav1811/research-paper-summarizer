"""
Arxiv Paper Fetcher

This module handles fetching research papers from the Arxiv API.
It provides functionality to search papers, parse XML responses, 
and extract paper metadata.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivPaperFetcher:
    """Fetches papers from Arxiv API with rate limiting and error handling"""
    
    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize the Arxiv fetcher
        
        Args:
            rate_limit_delay: Delay between requests to respect API limits
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def search_papers(self, query: str, max_results: int = 10, 
                     start: int = 0, sort_by: str = 'submittedDate', 
                     sort_order: str = 'descending') -> List[Dict]:
        """
        Search papers on Arxiv
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort criterion ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending', 'descending')
            
        Returns:
            List of paper dictionaries
        """
        # Respect rate limiting
        self._rate_limit()
        
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            logger.info(f"Fetching papers for query: '{query}' (max_results={max_results})")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            # Handle namespace
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', namespace)
            logger.info(f"Found {len(entries)} papers")
            
            for entry in entries:
                paper = self._parse_entry(entry, namespace)
                if paper:  # Only add valid papers
                    papers.append(paper)
            
            return papers
            
        except requests.RequestException as e:
            logger.error(f"Error fetching papers: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"Error parsing XML response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    
    def _parse_entry(self, entry, namespace: Dict[str, str]) -> Dict:
        """
        Parse individual paper entry from XML
        
        Args:
            entry: XML entry element
            namespace: XML namespace mapping
            
        Returns:
            Paper dictionary or None if parsing fails
        """
        try:
            # Extract basic information
            title_elem = entry.find('atom:title', namespace)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Unknown Title"
            
            summary_elem = entry.find('atom:summary', namespace)
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespace):
                name_elem = author.find('atom:name', namespace)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract publication date
            published_elem = entry.find('atom:published', namespace)
            published = published_elem.text if published_elem is not None else ""
            
            # Extract updated date
            updated_elem = entry.find('atom:updated', namespace)
            updated = updated_elem.text if updated_elem is not None else ""
            
            # Extract Arxiv ID
            id_elem = entry.find('atom:id', namespace)
            arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else ""
            
            # Extract links (PDF, abstract page)
            pdf_link = None
            abs_link = None
            
            for link in entry.findall('atom:link', namespace):
                link_type = link.get('type', '')
                href = link.get('href', '')
                
                if link_type == 'application/pdf':
                    pdf_link = href
                elif 'abs' in href:
                    abs_link = href
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespace):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract primary category
            primary_category = entry.find('arxiv:primary_category', namespace)
            primary_cat = primary_category.get('term') if primary_category is not None else ""
            
            # Extract journal reference if available
            journal_ref_elem = entry.find('arxiv:journal_ref', namespace)
            journal_ref = journal_ref_elem.text if journal_ref_elem is not None else ""
            
            # Extract DOI if available
            doi_elem = entry.find('arxiv:doi', namespace)
            doi = doi_elem.text if doi_elem is not None else ""
            
            paper_data = {
                'title': title,
                'authors': authors,
                'summary': summary,
                'published': published,
                'updated': updated,
                'arxiv_id': arxiv_id,
                'pdf_link': pdf_link,
                'abs_link': abs_link,
                'categories': categories,
                'primary_category': primary_cat,
                'journal_ref': journal_ref,
                'doi': doi
            }
            
            return paper_data
            
        except Exception as e:
            logger.warning(f"Failed to parse paper entry: {e}")
            return None
    
    def _rate_limit(self):
        """Implement rate limiting to respect Arxiv API guidelines"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_paper_by_id(self, arxiv_id: str) -> Dict:
        """
        Fetch a specific paper by Arxiv ID
        
        Args:
            arxiv_id: Arxiv paper ID (e.g., '2301.07041')
            
        Returns:
            Paper dictionary or empty dict if not found
        """
        query = f"id:{arxiv_id}"
        papers = self.search_papers(query, max_results=1)
        return papers[0] if papers else {}
    
    def search_by_author(self, author_name: str, max_results: int = 10) -> List[Dict]:
        """
        Search papers by author name
        
        Args:
            author_name: Author name to search
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"au:{author_name}"
        return self.search_papers(query, max_results=max_results)
    
    def search_by_category(self, category: str, max_results: int = 10) -> List[Dict]:
        """
        Search papers by Arxiv category
        
        Args:
            category: Arxiv category (e.g., 'cs.AI', 'cs.LG')
            max_results: Maximum number of results
            
        Returns:
            List of paper dictionaries
        """
        query = f"cat:{category}"
        return self.search_papers(query, max_results=max_results)