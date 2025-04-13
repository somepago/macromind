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
        predictions = [map_trend_to_numeric(p) for p in precomputed_predictions]
        sentiments = precomputed_sentiments
    else:
        predictions = []
        sentiments = []

        for commodity in commodities:
            stock_prompt = com_pred.format_stock_data(commodity, startdate)
            headlines = com_pred.format_news(commodity, startdate)
            prompt = com_pred.build_prompt(commodity, stock_prompt, headlines)
            raw_prediction = com_pred.predict_direction(prompt)
            prediction, _ = com_pred.parse_prediction(raw_prediction)
            sentiment_score = com_pred.compute_sentiment_score_bulk(headlines)

            predictions.append(map_trend_to_numeric(prediction))
            sentiments.append(sentiment_score)
    
    forecast_results = []
    start_date = pd.to_datetime(startdate)

    for idx, commodity in enumerate(commodities):
        sentiment_score = sentiments[idx]
        trend_prediction = predictions[idx]
        
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
        model.add_regressor('trend_prediction')  # Add trend prediction as a regressor
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
        forecast['trend_macro'] = trend_prediction
        forecast['gain_percent'] = ((forecast['yhat'] - current_price) / current_price) * 100
        
        forecast['Commodity'] = commodity
        forecast_results.append(forecast[['ds', 'yhat', 'Commodity', 'sentiment', 'trend_macro',
                                         'current_price', 'predicted_price', 'gain_percent']])

    # Combine all forecasts
    all_forecasts = pd.concat(forecast_results, ignore_index=True)
    return all_forecasts

def allocate_funds(forecast_df: pd.DataFrame, total_funds: float = 100) -> pd.DataFrame:
    """
    Allocates funds to different commodities based on predicted gain percentages.
    If the gain percentage is negative, the commodity is not allocated any funds.
    Also calculates the potential dollar gain.

    Parameters:
    - forecast_df: DataFrame with columns ['Commodity', 'current_price', 'predicted_price', 'gain_percent']
    - total_funds: Total available funds to allocate (default is $100)

    Returns:
    - DataFrame with allocation and potential dollar gain for each commodity
    """
    # Step 1: Calculate the gain percentage (if not already present)
    forecast_df['gain_percent'] = ((forecast_df['predicted_price'] - forecast_df['current_price']) / forecast_df['current_price']) * 100

    # Create a working copy that includes all commodities for display
    all_commodities_df = forecast_df.copy()
    
    # Step 2: Filter out commodities with negative gain_percent for allocation purposes
    positive_returns_df = forecast_df[forecast_df['gain_percent'] > 0].copy()

    # If there are no positive gain commodities, allocate based on least negative
    if positive_returns_df.empty:
        # For negative returns, find the least negative ones
        forecast_df['allocation'] = 0  # Default to zero allocation
        
        # Only if we must allocate, choose least negative options
        if len(forecast_df) > 0:
            # Shift values to make them positive for allocation
            min_gain = forecast_df['gain_percent'].min()
            if min_gain < 0:
                adjusted_gain = forecast_df['gain_percent'] - min_gain + 0.1  # Small positive offset
                total_adj_gain = adjusted_gain.sum()
                forecast_df['allocation'] = (adjusted_gain / total_adj_gain) * total_funds
            else:
                # Equal allocation if all are zero
                forecast_df['allocation'] = total_funds / len(forecast_df)
    else:
        # Step 3: Normalize positive gain percentages to sum to 100%
        total_gain = positive_returns_df['gain_percent'].sum()
        positive_returns_df['allocation'] = (positive_returns_df['gain_percent'] / total_gain) * total_funds
        
        # Merge allocations back to original dataframe
        forecast_df = forecast_df.copy()
        forecast_df['allocation'] = 0  # Default all to zero
        for idx, row in positive_returns_df.iterrows():
            forecast_df.loc[forecast_df['Commodity'] == row['Commodity'], 'allocation'] = row['allocation']

    # Step 5: Calculate the potential dollar gain for each commodity
    forecast_df['predicted_gain_dollars'] = (
        (forecast_df['predicted_price'] - forecast_df['current_price']) *
        (forecast_df['allocation'] / forecast_df['current_price'])
    )
    
    forecast_df['MacroEconIndicator'] = forecast_df['trend_macro']
    forecast_df['Sentiment Score'] = forecast_df['sentiment']
    forecast_df.rename(columns={'gain_percent': 'Predicted Gain (%)', 'allocation': 'Allocation (%)', 'predicted_gain_dollars':'Projected Return ($)'}, inplace=True)
    forecast_df = forecast_df[['Commodity', 'MacroEconIndicator', 'Sentiment Score', 'Predicted Gain (%)', 'Allocation (%)', 'Projected Return ($)']]
    return forecast_df


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

    allocations = allocate_funds(forecast_df,100)
    print(allocations)
