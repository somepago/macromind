
HISTORY_IN_DAYS = 60

current_news_data = "data_prep/scraped_news_finhub/commodity_news_20250411_230516.csv"
current_stock_data = "data_prep/commodity_data_60days.csv"

# Commodity to ETF/Stock mapping
commodity_to_etfs_stocks = {
    'gold': {
        'ETFs': ['GLD', 'GDX', 'IAU'],
        'Stocks': ['GOLD', 'NEM', 'AEM']
    },
    'silver': {
        'ETFs': ['SLV', 'SIVR'],
        'Stocks': ['PAAS', 'AG', 'WPM']
    },
    'oil': {
        'ETFs': ['USO', 'XLE', 'OIH'],
        'Stocks': ['XOM', 'CVX', 'SLB', 'HAL']
    },
    'natural gas': {
        'ETFs': ['UNG', 'KOLD'],
        'Stocks': ['LNG', 'COG', 'EQT']
    },
    'wheat': {
        'ETFs': ['WEAT', 'CORN'],
        'Stocks': ['ADM', 'BG']
    },
    'corn': {
        'ETFs': ['CORN'],
        'Stocks': ['DE', 'ADM']
    },
    'lithium': {
        'ETFs': ['LIT', 'BATT'],
        'Stocks': ['ALB', 'SQM', 'LTHM']
    },
    'cobalt': {
        'ETFs': ['COBALT'],
        'Stocks': ['GLEN', 'CMOC', 'VALE']
    },
    'nickel': {
        'ETFs': ['NICKEL'],
        'Stocks': ['GMKN', 'VALE', 'AAL']
    },
    'coffee': {
        'ETFs': ['JO', 'CUP', 'CAFE'],
        'Stocks': ['SBUX', 'TATYF', 'LMWRF', 'MCD', 'KHC']
    }

}


def ticker_to_commodity_mapping(com_to_tick_dict):
    mapping_dict = {}
    for key, value in com_to_tick_dict.items():
        for ticker in value["ETFs"]:
            mapping_dict[ticker] = key

        for ticker in value["Stocks"]:
            mapping_dict[ticker] = key

    return mapping_dict

etfs_stocks_to_commodity = ticker_to_commodity_mapping(commodity_to_etfs_stocks)