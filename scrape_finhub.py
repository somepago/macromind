import time
import requests
import pandas as pd
import datetime
import os
from constants import commodity_to_etfs_stocks, HISTORY_IN_DAYS
# Your Finnhub API key
api_key = "cvrmao1r01qnpem8bhi0cvrmao1r01qnpem8bhig"

# Finnhub news endpoint
base_url = "https://finnhub.io/api/v1/company-news"

# Date range: last 60 days
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=HISTORY_IN_DAYS)
from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

# Get a list of all unique tickers
def get_all_tickers(commodity_mapping):
    tickers = set()
    for assets in commodity_mapping.values():
        tickers.update(assets['ETFs'])
        tickers.update(assets['Stocks'])
    return list(tickers)

# Fetch news for each valid ticker symbol
def fetch_news_by_ticker(tickers, from_date, to_date, api_key):
    all_articles = []
    rate_limit = 30
    interval = 1 / rate_limit

    for ticker in tickers:
        url = f"{base_url}?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            for article in response.json():
                article['ticker'] = ticker
                all_articles.append(article)
        else:
            print(f"Error fetching news for ticker '{ticker}': {response.status_code}")
        
        time.sleep(interval)

    return all_articles

# Optional: map each article to a commodity based on its ticker
def map_articles_to_commodities(articles, mapping):
    commodity_articles = []
    ticker_to_commodity = {}

    for commodity, assets in mapping.items():
        for ticker in assets['ETFs'] + assets['Stocks']:
            ticker_to_commodity[ticker] = commodity

    for article in articles:
        ticker = article.get('ticker')
        commodity = ticker_to_commodity.get(ticker)
        if commodity:
            article['commodity'] = commodity
            commodity_articles.append(article)

    return commodity_articles

# Run the pipeline
tickers = get_all_tickers(commodity_to_etfs_stocks)
news = fetch_news_by_ticker(tickers, from_date, to_date, api_key)
if news:
    mapped_news = map_articles_to_commodities(news, commodity_to_etfs_stocks)
    df = pd.DataFrame(mapped_news)
    print(df[['datetime', 'headline', 'source', 'url', 'ticker', 'commodity']].head())
    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data_prep/scraped_news_finhub/commodity_news_{timestamp}.csv"
    # Create directory if it doesn't exist
    os.makedirs('data_prep/scraped_news_finhub', exist_ok=True)
    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"News data saved to {output_file}")
    
else:
    print("No news articles found.")
