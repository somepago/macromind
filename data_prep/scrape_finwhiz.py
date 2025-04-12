from bs4 import BeautifulSoup
import requests
import csv
from datetime import datetime
import time

def scrape_finviz_news():
    """
    Scrape news headlines from Finviz.
    Returns a list of tuples containing (title, link, datetime).
    """
    url = 'https://finviz.com/news.ashx'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = []
        
        # Find all news cells with the correct class
        news_cells = soup.find_all('td', class_='news_link-cell')
        
        for cell in news_cells:
            try:
                # Find the link within the cell
                link_elem = cell.find('a', class_='nn-tab-link')
                if link_elem:
                    title = link_elem.text.strip()
                    link = link_elem['href']
                    
                    # Try to get datetime from the data-boxover attribute
                    datetime_str = None
                    if 'data-boxover' in cell.attrs:
                        # Extract datetime from the tooltip if available
                        tooltip_text = cell['data-boxover']
                        if 'datetime=' in tooltip_text:
                            datetime_str = tooltip_text.split('datetime=')[1].split(']')[0]
                    
                    if title and link:
                        headlines.append((title, link, datetime_str))
            except Exception as e:
                print(f"Error processing news cell: {e}")
                continue
        
        if not headlines:
            print("Warning: No headlines found. Response content:")
            print(res.text[:500])
            
        return headlines
        
    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def get_filtered_news(keywords):
    """
    Get news headlines filtered by specific keywords.
    
    Args:
        keywords (list): List of keywords to filter news by
        
    Returns:
        list: List of tuples containing (title, link, datetime, matched_keywords) for matching headlines
    """
    all_headlines = scrape_finviz_news()
    
    if not all_headlines:
        print("No headlines were retrieved. Please check the scraping function.")
        return []
        
    filtered_headlines = []
    
    for title, link, datetime_str in all_headlines:
        matched_keywords = [keyword for keyword in keywords if keyword.lower() in title.lower()]
        if matched_keywords:
            filtered_headlines.append((title, link, datetime_str, matched_keywords))
    
    return filtered_headlines

def save_to_csv(headlines, filename=None):
    """
    Save the filtered headlines to a CSV file.
    
    Args:
        headlines (list): List of tuples containing (title, link, datetime, matched_keywords)
        filename (str, optional): Name of the CSV file. If None, generates a timestamp-based name.
    """
    if not headlines:
        print("No headlines to save to CSV.")
        return None
        
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"commodity_news_{timestamp}.csv"
    
    try:
        with open(f"scraped_data/{filename}", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Link', 'DateTime', 'Matched Keywords'])
            for title, link, datetime_str, keywords in headlines:
                writer.writerow([title, link, datetime_str or 'N/A', ', '.join(keywords)])
        return filename
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return None

if __name__ == "__main__":
    # Example keywords
    keywords = ["oil","gold"]
    
    print("Fetching news...")
    filtered_news = get_filtered_news(keywords)
    
    if filtered_news:
        # Save to CSV
        csv_file = save_to_csv(filtered_news)
        if csv_file:
            print(f"News saved to {csv_file}")
        


