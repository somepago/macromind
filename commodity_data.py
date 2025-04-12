from constants import commodity_to_etfs_stocks,etfs_stocks_to_commodity
import yfinance as yf
import pandas as pd
from scrape_finhub import get_all_tickers

def commodity_to_tickers(commodity, ticker_type="stock"):
    """
    Function to recommend ETFs and stocks based on the commodity.
    
    Arguments:
    commodity -- The commodity name (e.g., 'gold', 'oil', etc.)
    
    Returns:
    List of recommended ETFs and stocks
    """
    commodity_mapping = commodity_to_etfs_stocks.get(commodity.lower())
    
    if commodity_mapping:
        if ticker_type=="stock":
            return commodity_mapping["ETFs"]
        elif ticker_type=="etf":
            return commodity_mapping["Stocks"]
        else:
            raise Exception("Incorrect ticker type")
    else:
        return None
    


# Function to fetch historical prices for the last 'num_days' days and calculate daily average
def fetch_daily_average_price(tickers, num_days=5):
    data = yf.download(tickers, period=f"{num_days}d", interval="1d", group_by='ticker', auto_adjust=True)

    # Compute average daily price = (Open + Close) / 2
    avg_price_df = pd.DataFrame()
    for ticker in tickers:
        if ticker in data.columns.levels[0]:
            ticker_data = data[ticker]
            ticker_data['Average'] = (ticker_data['Open'] + ticker_data['Close']) / 2
            ticker_data['Ticker'] = ticker
            ticker_data["Commodity"] = etfs_stocks_to_commodity[ticker]
            avg_price_df = pd.concat([avg_price_df, ticker_data[['Average', 'Ticker','Commodity']].copy()])

    avg_price_df.reset_index(inplace=True)

    print(avg_price_df.head())
    
    return avg_price_df

# Example usage: Fetch the last 5 days of average stock prices
stock_tickers = get_all_tickers(commodity_to_etfs_stocks)
df_stock_prices_avg = fetch_daily_average_price(stock_tickers, num_days=60)
df_stock_prices_avg.to_csv("data_prep/commodity_data_60days.csv")
# Display the fetched data
print(df_stock_prices_avg)
