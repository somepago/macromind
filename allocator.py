from predictor import CommodityPricePredictor
from typing import List, Callable, Literal
import pandas as pd
import numpy as np
from constants import current_news_data, current_stock_data

def softmax(x, temp=1.0):
    x = np.array(x)
    e_x = np.exp(x / temp)
    return (e_x / e_x.sum()).tolist()

# Define strategy mapping
def get_weighting_function(strategy: str) -> Callable[[str, float], float]:
    """
    Returns a function (prediction, sentiment) -> score, based on the strategy.
    """
    strategies = {
        "linear": lambda pred, s: {
            "UP": s,
            "SAME": 0.3 * s,
            "DOWN": 0.0
        }.get(pred, 0.0),

        "aggressive": lambda pred, s: {
            "UP": 0.5 + 0.5 * s,
            "SAME": 0.1 + 0.2 * s,
            "DOWN": 0.0
        }.get(pred, 0.0),

        "conservative": lambda pred, s: {
            "UP": 0.2 + 0.3 * s,
            "SAME": 0.1 * s,
            "DOWN": 0.0
        }.get(pred, 0.0),

        "neutral_aware": lambda pred, s: {
            "UP": s,
            "SAME": s * 0.1,
            "DOWN": (1 - s) * 0.05  # Very small allocation for low sentiment & down
        }.get(pred, 0.0)
    }

    return strategies.get(strategy, strategies["linear"])  # Default to "linear"

def commodity_allocator(com_pred: CommodityPricePredictor,
                        commodities: List[str],
                        startdate: str,
                        budget: float = 100.0,
                        assistant_mode: str = "verbose",
                        strategy: str = "aggressive",
                        temp: float = 0.5,
                        precomputed_predictions: List[str] = None,
                        precomputed_sentiments: List[float] = None) -> pd.DataFrame:
    """
    Allocate portfolio dollar amounts across commodities based on LLM predictions and sentiment.

    Parameters:
    - com_pred: CommodityPricePredictor instance
    - commodities: List of commodity names
    - startdate: Date string (e.g., "2025-04-06")
    - budget: Total dollars to allocate (default is 100)
    - assistant_mode: "verbose" or "terse"
    - strategy: Weighting strategy ("linear", "aggressive", "conservative", etc.)
    - temp: Temperature for softmax (lower = sharper differences)
    - precomputed_predictions: Optional list of precomputed predictions (UP, DOWN, SAME)
    - precomputed_sentiments: Optional list of precomputed sentiment scores (0 to 1)

    Returns a DataFrame with:
    - Commodity
    - Prediction (UP, DOWN, SAME)
    - Sentiment Score (0 to 1)
    - Allocation Weight (0 to 1)
    - Dollar Allocation (rounded to 2 decimals)
    """
    # Use precomputed values if provided, otherwise compute them
    if precomputed_predictions is not None and precomputed_sentiments is not None:
        predictions = [p.upper() for p in precomputed_predictions]
        sentiments = precomputed_sentiments
    else:
        predictions = []
        sentiments = []

        for commodity in commodities:
            stock_prompt = com_pred.format_stock_data(commodity, startdate)
            headlines = com_pred.format_news(commodity, startdate)
            prompt = com_pred.build_prompt(commodity, stock_prompt, headlines, assistant_mode)
            raw_prediction = com_pred.predict_direction(prompt)
            prediction, _ = com_pred.parse_prediction(raw_prediction)
            sentiment_score = com_pred.compute_sentiment_score_bulk(headlines)

            predictions.append(prediction.upper())
            sentiments.append(sentiment_score)

    # Get the strategy-based scoring function
    weight_fn = get_weighting_function(strategy)
    raw_scores = [weight_fn(pred, sentiment) for pred, sentiment in zip(predictions, sentiments)]

    weights = softmax(raw_scores, temp=temp)
    allocations = [round(w * budget, 2) for w in weights]

    allocation_df = pd.DataFrame({
        "Commodity": commodities,
        "Prediction": predictions,
        "Sentiment Score": [round(s, 3) for s in sentiments],
        "Allocation Weight": [round(w, 3) for w in weights],
        "Dollar Allocation": allocations
    })

    return allocation_df


def project_gains(allocation_df: pd.DataFrame,
                  price_df: str,
                  startdate: str,
                  horizon_days: int = 1,
                  mode: Literal["backtest", "forecast"] = "backtest") -> pd.DataFrame:
    """
    Project gains based on dollar allocation and either backtested or forecasted prices.

    Parameters:
    - allocation_df: Output DataFrame from commodity_allocator
    - price_df: Historical price DataFrame (must include 'Date', 'Commodity', 'Average')
    - startdate: Start date of the investment (string format: "YYYY-MM-DD")
    - horizon_days: How far to look ahead for gain (1 = next day, 30 = a month)
    - mode: "backtest" uses actual prices; "forecast" uses simple moving average

    Returns a new DataFrame with projected gains.
    """
    price_df = pd.read_csv(price_df)
    startdate = pd.to_datetime(startdate)
    future_date = startdate + pd.Timedelta(days=horizon_days)

    result = []

    for _, row in allocation_df.iterrows():
        commodity = row['Commodity']
        allocation = row['Dollar Allocation']

        df = price_df[price_df['Commodity'].str.lower() == commodity.lower()].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Get current price (price at or before startdate)
        past_price_row = df[df['Date'] <= startdate].sort_values(by='Date', ascending=False).head(1)
        if past_price_row.empty:
            continue
        current_price = past_price_row['Average'].values[0]

        if mode == "backtest":
            # Use actual future price
            future_price_row = df[df['Date'] >= future_date].sort_values(by='Date').head(1)
            if future_price_row.empty:
                continue
            future_price = future_price_row['Average'].values[0]
        elif mode == "forecast":
            # Forecast using simple moving average of previous 10 days
            window = df[df['Date'] < startdate].tail(10)['Average']
            future_price = window.mean()
        else:
            raise ValueError("mode must be 'backtest' or 'forecast'")

        pct_gain = (future_price - current_price) / current_price
        projected_return = allocation * pct_gain

        result.append({
            "Commodity": commodity,
            "Current Price": round(current_price, 2),
            "Future Price": round(future_price, 2),
            "Dollar Allocation": allocation,
            "Gain (%)": round(pct_gain * 100, 2),
            "Projected Return ($)": round(projected_return, 2),
            "Mode": mode,
            "Horizon (days)": horizon_days
        })

    return pd.DataFrame(result)


if __name__ == "__main__":
    com_pred = CommodityPricePredictor(news_df=current_news_data, price_df=current_stock_data)
    commodities = ["coffee", "gold", "oil"]
    startdate = "2025-04-06"
    budget = 100.0

    allocation = commodity_allocator(com_pred, commodities, startdate, budget)

    # Backtest gain (e.g., 5 days later)
    gains_bt = project_gains(allocation, current_stock_data, startdate, horizon_days=5, mode="backtest")
    print("\n--- Backtest Gains ---")
    print(gains_bt)

    # Forecast gain
    gains_forecast = project_gains(allocation, current_stock_data, startdate, horizon_days=5, mode="forecast")
    print("\n--- Forecast Gains ---")
    print(gains_forecast)