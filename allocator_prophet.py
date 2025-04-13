from prophet import Prophet
import pandas as pd
from typing import List, Optional
from predictor import CommodityPricePredictor

# Mapping multiclass trend predictions to numerical values
def map_trend_to_numeric(trend: str) -> int:
    trend_map = {
        'UP': 1,
        'DOWN': -1,
        'SAME': 0,
        'MODERATE_UP': 0.5,
        'MODERATE_DOWN': -0.5
    }
    return trend_map.get(trend.upper(), 0)  # Default to 'SAME' if unknown trend

# Enhanced mapping function with boosted weights for macro trends
def map_trend_to_numeric_boosted(trend: str, boost_factor: float = 2.5) -> float:
    """Maps trend predictions to numeric values with a boost factor to increase their influence"""
    base_value = map_trend_to_numeric(trend)
    return base_value * boost_factor  # Apply boosting to increase influence

def price_forecast(
    com_pred: CommodityPricePredictor,
    price_df: str, 
    commodities: List[str], 
    startdate: str,
    forecast_days: int = 1,
    precomputed_predictions: Optional[List[str]] = None,
    precomputed_sentiments: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Forecast future prices using Prophet for multiple commodities, including sentiment and trend predictions as regressors.

    Parameters:
    - com_pred: Instance of CommodityPricePredictor
    - price_df: DataFrame with ['Date', 'Commodity', 'Average']
    - commodities: List of commodity names
    - startdate: Date string (e.g., "2025-04-06")
    - forecast_days: Number of days to forecast into the future
    - precomputed_predictions: Optional list of trend predictions (UP/DOWN/SAME)
    - precomputed_sentiments: Optional list of sentiment scores

    Returns:
    - Combined forecast DataFrame for all commodities
    """
    price_df = pd.read_csv(price_df)
    
    # Use precomputed values if provided, otherwise compute them
    if precomputed_predictions is not None and precomputed_sentiments is not None:
        original_predictions = precomputed_predictions.copy()  # Store original text predictions
        predictions = [map_trend_to_numeric_boosted(p) for p in precomputed_predictions]
        sentiments = precomputed_sentiments
    else:
        predictions = []
        original_predictions = []  # Store original text predictions
        sentiments = []

        for commodity in commodities:
            stock_prompt = com_pred.format_stock_data(commodity, startdate)
            headlines = com_pred.format_news(commodity, startdate)
            prompt = com_pred.build_prompt(commodity, stock_prompt, headlines)
            raw_prediction = com_pred.predict_direction(prompt)
            prediction, _ = com_pred.parse_prediction(raw_prediction)
            sentiment_score = com_pred.compute_sentiment_score_bulk(headlines)

            original_predictions.append(prediction.upper())  # Store original text prediction
            predictions.append(map_trend_to_numeric_boosted(prediction))
            sentiments.append(sentiment_score)
    
    forecast_results = []
    start_date = pd.to_datetime(startdate)

    for idx, commodity in enumerate(commodities):
        sentiment_score = sentiments[idx]
        trend_prediction = predictions[idx]
        trend_text = original_predictions[idx]  # Get original text prediction
        
        # Prepare price data
        price_grouped = price_df.groupby(['Commodity', 'Date'], as_index=False)['Average'].mean()
        
        df_price = price_grouped[
            (price_grouped['Commodity'].str.lower() == commodity.lower()) & 
            (pd.to_datetime(price_grouped['Date']) <= start_date)
        ].copy()

        df_price['ds'] = pd.to_datetime(df_price['Date'])
        df_price = df_price[['ds', 'Average']].rename(columns={'Average': 'y'})
        df_price['sentiment_score'] = sentiment_score  # Constant for all past rows
        df_price['trend_prediction'] = trend_prediction  # Add trend prediction as regressor

        if df_price.empty or df_price.shape[0] < 5:
            print(f"[Warning] Skipping {commodity} due to insufficient data.")
            continue

        # Fit Prophet model with sentiment and trend prediction
        model = Prophet()
        model.add_regressor('sentiment_score')
        model.add_regressor('trend_prediction', standardize=False)  # Don't standardize to preserve boosted impact
        model.fit(df_price)

        # Create future frame
        future = model.make_future_dataframe(periods=forecast_days)
        future['sentiment_score'] = sentiment_score  # Extend with same sentiment
        future['trend_prediction'] = trend_prediction  # Extend with same trend prediction

        # Predict
        forecast = model.predict(future)
        
        # Only keep the last day (forecast day)
        target_date = df_price['ds'].max() + pd.Timedelta(days=forecast_days)
        forecast = forecast[forecast['ds'] == target_date]
        
        # Get current price (latest available price)
        current_price = df_price.iloc[-1]['y']
        
        # Add current price and calculate gain
        forecast['current_price'] = current_price
        forecast['predicted_price'] = forecast['yhat']
        forecast['sentiment'] = sentiment_score
        forecast['trend_macro'] = trend_text  # Store original text prediction
        forecast['trend_numeric'] = trend_prediction  # Keep numeric version for calculations
        forecast['gain_percent'] = ((forecast['yhat'] - current_price) / current_price) * 100
        
        forecast['Commodity'] = commodity
        forecast_results.append(forecast[['ds', 'yhat', 'Commodity', 'sentiment', 'trend_macro',
                                         'trend_numeric', 'current_price', 'predicted_price', 'gain_percent']])

    # Combine all forecasts
    all_forecasts = pd.concat(forecast_results, ignore_index=True)
    return all_forecasts

def allocate_funds(forecast_df: pd.DataFrame, total_funds: float = 100, strategy: str = 'aggressive') -> pd.DataFrame:
    """
    Allocates funds to different commodities based on predicted gain percentages.
    If the gain percentage is negative, the commodity is not allocated any funds.
    Also calculates the potential dollar gain.

    Parameters:
    - forecast_df: DataFrame with columns ['Commodity', 'current_price', 'predicted_price', 'gain_percent']
    - total_funds: Total available funds to allocate (default is $100)
    - strategy: Investment strategy to use ('aggressive', 'conservative', 'neutral_aware', 'equal')
      - aggressive: Heavily weights commodities with highest predicted gains
      - conservative: More cautious allocation including less negative options
      - neutral_aware: Considers both gain percentage and sentiment/trend data
      - equal: Equal allocation to commodities with positive gains

    Returns:
    - DataFrame with allocation and potential dollar gain for each commodity
    """
    # Make a copy to avoid modifying the original
    forecast_df = forecast_df.copy()
    
    # Step 1: Calculate the gain percentage (if not already present)
    if 'gain_percent' not in forecast_df.columns:
        forecast_df['gain_percent'] = ((forecast_df['predicted_price'] - forecast_df['current_price']) / forecast_df['current_price']) * 100

    # Initialize allocation column
    forecast_df['allocation'] = 0
    
    # Apply different allocation strategies
    if strategy == 'aggressive':
        # Aggressive: Square the positive gain percentages to amplify differences
        positive_returns_df = forecast_df[forecast_df['gain_percent'] > 0].copy()
        if not positive_returns_df.empty:
            # Square the gain percentages to give much higher weight to higher gains
            positive_returns_df['weight'] = positive_returns_df['gain_percent'] ** 2
            total_weight = positive_returns_df['weight'].sum()
            
            # Update allocations for positive return commodities in the original dataframe
            for idx, row in positive_returns_df.iterrows():
                allocation_amount = (row['weight'] / total_weight) * total_funds
                forecast_df.loc[idx, 'allocation'] = allocation_amount
        else:
            # If no positive returns, allocate to the least negative
            least_negative_idx = forecast_df['gain_percent'].idxmax()
            forecast_df.loc[least_negative_idx, 'allocation'] = total_funds
    
    elif strategy == 'conservative':
        # Conservative: Consider all commodities but skew toward less negative ones
        # Shift all gain percentages to make them relative to the worst performer
        min_gain = forecast_df['gain_percent'].min() - 0.1  # Subtract 0.1 to ensure all values are positive
        forecast_df['shifted_gain'] = forecast_df['gain_percent'] - min_gain
        
        # Use the square root to reduce differences (more equal allocation)
        forecast_df['weight'] = forecast_df['shifted_gain'].apply(lambda x: max(0.1, x ** 0.5))
        total_weight = forecast_df['weight'].sum()
        
        # Allocate proportionally
        forecast_df['allocation'] = (forecast_df['weight'] / total_weight) * total_funds
    
    elif strategy == 'neutral_aware':
        # Neutral aware: Consider both gain percentage and sentiment
        # Create a combined score using gain percentage and sentiment
        if 'sentiment' in forecast_df.columns:
            forecast_df['combined_score'] = forecast_df['gain_percent'] + (forecast_df['sentiment'] * 20)
            
            # Add a small boost based on the trend direction
            if 'trend_numeric' in forecast_df.columns:
                # Using a gentler weight of 2 for trend impact
                forecast_df['combined_score'] += forecast_df['trend_numeric'] * 2
            
            # Ensure minimum score is positive
            min_score = forecast_df['combined_score'].min() - 0.1
            if min_score < 0:
                forecast_df['adjusted_score'] = forecast_df['combined_score'] - min_score
            else:
                forecast_df['adjusted_score'] = forecast_df['combined_score']
                
            # Allocate based on adjusted score
            total_score = forecast_df['adjusted_score'].sum()
            forecast_df['allocation'] = (forecast_df['adjusted_score'] / total_score) * total_funds
        else:
            # Fall back to equal if sentiment data not available
            # Only distribute among positive gain commodities
            positive_returns_df = forecast_df[forecast_df['gain_percent'] > 0]
            if not positive_returns_df.empty:
                equal_allocation = total_funds / len(positive_returns_df)
                for idx in positive_returns_df.index:
                    forecast_df.loc[idx, 'allocation'] = equal_allocation
    
    else:  # 'equal' strategy (default)
        # Equal: Only distribute among positive gain commodities
        positive_returns_df = forecast_df[forecast_df['gain_percent'] > 0]
        if not positive_returns_df.empty:
            equal_allocation = total_funds / len(positive_returns_df)
            for idx in positive_returns_df.index:
                forecast_df.loc[idx, 'allocation'] = equal_allocation
        elif len(forecast_df) > 0:
            # If no positive returns, allocate to the least negative
            least_negative_idx = forecast_df['gain_percent'].idxmax()
            forecast_df.loc[least_negative_idx, 'allocation'] = total_funds
    
    # Step 3: Calculate the potential dollar gain for each commodity
    forecast_df['predicted_gain_dollars'] = (
        (forecast_df['predicted_price'] - forecast_df['current_price']) *
        (forecast_df['allocation'] / forecast_df['current_price'])
    )
    
    # Prepare the display columns
    forecast_df['MacroEconIndicator'] = forecast_df['trend_macro']
    forecast_df['Sentiment Score'] = forecast_df['sentiment']
    forecast_df.rename(columns={
        'gain_percent': 'Predicted Gain (%)', 
        'allocation': 'Allocation (%)', 
        'predicted_gain_dollars': 'Projected Return ($)'
    }, inplace=True)
    
    # Select and order the columns for display
    result_df = forecast_df[['Commodity', 'MacroEconIndicator', 'Sentiment Score', 
                            'Predicted Gain (%)', 'Allocation (%)', 'Projected Return ($)']]
    
    return result_df

if __name__ == "__main__":
    from constants import current_news_data, current_stock_data
    commodities = ["gold", "oil"]
    startdate = "2025-02-12"
    com_pred = CommodityPricePredictor(news_df=current_news_data, price_df=current_stock_data)

    forecast_df = price_forecast(
        com_pred=com_pred,
        price_df=current_stock_data,
        commodities=commodities,
        startdate=startdate,
        forecast_days=1,

    )

    print(forecast_df)

    allocations = allocate_funds(forecast_df,100, strategy='aggressive')
    print(allocations)
