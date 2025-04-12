import streamlit as st
from main import CommodityPricePredictor  # Replace 'your_module' with actual filename (without .py)
from datetime import datetime
import pandas as pd
from constants import current_news_data, current_stock_data, predictable_commodities

# Load data just once
@st.cache_data
def load_data():
    price_df = pd.read_csv(current_stock_data)
    news_path = current_news_data
    com_pred = CommodityPricePredictor(news_df=news_path, price_df="data_prep/commodity_data_60days.csv")
    available_commodities = predictable_commodities
    dates = pd.to_datetime(price_df['Date']).dt.strftime('%Y-%m-%d').unique().tolist()
    return com_pred, available_commodities, dates

com_pred, commodities, available_dates = load_data()

# UI
st.title("📈🌎 MacroMind")

startdate = st.selectbox("Select a start date", sorted(available_dates, reverse=True))
assistant_mode = st.selectbox("Assistant Mode", ["crisp","verbose"])

if st.button("Predict Direction for All Commodities"):
    with st.spinner("Predicting for all commodities..."):
        # Create a container for predictions
        prediction_container = st.container()
        
        # Create columns for better organization
        with prediction_container:
            st.subheader("Predictions for All Commodities")
            for commodity in commodities:
                prediction, explanation = com_pred(commodity, startdate, "verbose")
                st.write(f"📊 **{commodity}**: {prediction}")

