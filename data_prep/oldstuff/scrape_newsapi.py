# Get news in last 60 days
# use newsapi.org
import requests
import pandas as pd
from datetime import datetime, timedelta
import os


def save_news_to_csv(num_days):
    # get news from newsapi.org
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError("NEWS_API_KEY environment variable not set")
    
    # Define keywords for commodity-related news
    keywords = [
    'gold', 'silver', 'platinum', 'palladium', 'precious metals', 'bullion',
    'oil', 'crude', 'brent', 'WTI', 'OPEC', 'petroleum', 'natural gas', 'LNG', 'energy prices', 'refinery', 'shale',
    'wheat', 'corn', 'soybeans', 'grain', 'coffee', 'cocoa', 'cotton', 'sugar', 'crop yield', 'harvest', 'farming', 'drought',
    'lithium', 'cobalt', 'nickel', 'copper', 'zinc', 'rare earth', 'battery metals', 'EV supply chain', 'mining',
    'commodity prices', 'commodity rally', 'inflation', 'supply chain', 'geopolitical tensions', 'sanctions',
    'rate hike', 'dollar strength', 'energy crisis', 'global demand', 'export ban'
]

    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Base URL for NewsAPI
    base_url = 'https://newsapi.org/v2/everything'
    
    all_articles = []
    
    # Fetch news for each keyword
    for keyword in keywords:
        params = {
            'q': keyword,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': api_key
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'ok':
                articles = data['articles']
                for article in articles:
                    article['keyword'] = keyword
                    all_articles.append(article)
        else:
            print(f"Error fetching news for keyword '{keyword}': {response.status_code}")
    
    # Convert to DataFrame
    if all_articles:
        df = pd.DataFrame(all_articles)
        
        # Select relevant columns
        columns = ['title', 'description', 'url', 'publishedAt', 'source', 'keyword']
        df = df[columns]
        
        # Rename columns for clarity
        df = df.rename(columns={
            'title': 'Title',
            'description': 'Description',
            'url': 'URL',
            'publishedAt': 'Published Date',
            'source': 'Source',
            'keyword': 'Keyword'
        })
        
        # Save to CSV
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'commodity_news_{num_days}days.csv')
        df.to_csv(output_file, index=False)
        print(f"News data saved to {output_file}")
        return df
    else:
        print("No articles found")
        return None

if __name__ == "__main__":
    # Example usage: get news for the last 60 days
    save_news_to_csv(60)
