from predictor import CommodityPricePredictor
from typing import List
import pandas as pd
from constants import current_news_data, current_stock_data

def commodity_allocator(com_pred: CommodityPricePredictor,
                        commodities: List[str],
                        startdate: str,
                        budget: float = 100.0,
                        assistant_mode: str = "verbose") -> pd.DataFrame:
    """
    Allocate portfolio dollar amounts across commodities based on LLM predictions and sentiment.

    Parameters:
    - com_pred: Instance of CommodityPricePredictor
    - commodities: List of commodity names
    - startdate: Date string (e.g., "2025-04-06")
    - budget: Total dollars to allocate (default is 100)
    - assistant_mode: Whether to use verbose or terse mode for predictions

    Returns a DataFrame with:
    - Commodity
    - Prediction (UP, DOWN, SAME)
    - Sentiment Score (0 to 1)
    - Allocation Weight (0 to 1)
    - Dollar Allocation (rounded to 2 decimals)
    """
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

    # Compute base scores
    base_scores = []
    for pred, sentiment in zip(predictions, sentiments):
        if pred == "UP":
            base_scores.append(sentiment)
        elif pred == "SAME":
            base_scores.append(sentiment * 0.3)  # Smaller allocation
        else:  # DOWN
            base_scores.append(0.0)

    # Normalize weights
    total_score = sum(base_scores)
    if total_score > 0:
        weights = [score / total_score for score in base_scores]
    else:
        # Fall back to equal allocation
        weights = [1 / len(commodities)] * len(commodities)

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
    allocation = commodity_allocator(com_pred, commodities, startdate, budget)
    print(allocation)
