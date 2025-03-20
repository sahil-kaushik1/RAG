import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def get_website_text_content(url):
    """
    Extract the main text content from a website URL.
    
    Args:
        url: The URL of the website to scrape
        
    Returns:
        The extracted text content
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            # Extract the main content
            text = trafilatura.extract(downloaded)
            if text:
                return text
        
        # Fallback to BeautifulSoup if trafilatura fails
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            return text
        
        return None
    except Exception as e:
        print(f"Error scraping website: {str(e)}")
        return None

def crawl_website(url, max_pages=10, max_depth=2):
    """
    Crawl a website and extract text from multiple pages
    
    Args:
        url: The starting URL
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth to crawl
        
    Returns:
        Dictionary mapping URLs to their extracted text
    """
    visited = set()
    to_visit = [(url, 0)]  # (url, depth)
    results = {}
    
    base_domain = urlparse(url).netloc
    
    while to_visit and len(visited) < max_pages:
        current_url, depth = to_visit.pop(0)
        
        if current_url in visited:
            continue
        
        visited.add(current_url)
        
        # Extract text from the current page
        text = get_website_text_content(current_url)
        if text:
            results[current_url] = text
        
        # Don't go deeper if we've reached max depth
        if depth >= max_depth:
            continue
        
        # Find links on the current page
        try:
            response = requests.get(current_url, headers={"User-Agent": "Mozilla/5.0"})
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Resolve relative URLs
                    full_url = urljoin(current_url, href)
                    
                    # Only follow links to the same domain
                    if urlparse(full_url).netloc == base_domain and full_url not in visited:
                        to_visit.append((full_url, depth + 1))
        except Exception as e:
            print(f"Error crawling {current_url}: {str(e)}")
    
    return results
