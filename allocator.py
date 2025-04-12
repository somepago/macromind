from predictor import CommodityPricePredictor
from typing import List, Callable
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

if __name__ == "__main__":
    com_pred = CommodityPricePredictor(news_df=current_news_data, price_df=current_stock_data)
    commodities = ["coffee", "gold", "oil"]
    startdate = "2025-04-06"
    budget = 100.0
    strategy = "aggressive"  # Try "conservative", "linear", etc.
    allocation = commodity_allocator(com_pred, commodities, startdate, budget, strategy=strategy)
    print(allocation)
