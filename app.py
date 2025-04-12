import streamlit as st
from main import CommodityPricePredictor  # Replace 'your_module' with actual filename (without .py)
from datetime import datetime
import pandas as pd
from constants import current_news_data, current_stock_data

# Load data just once
@st.cache_data
def load_data():
    price_df = pd.read_csv(current_stock_data)
    news_path = current_news_data
    com_pred = CommodityPricePredictor(news_df=news_path, price_df="data_prep/commodity_data_60days.csv")
    available_commodities = price_df['Commodity'].unique().tolist()
    dates = pd.to_datetime(price_df['Date']).dt.strftime('%Y-%m-%d').unique().tolist()
    return com_pred, available_commodities, dates

com_pred, commodities, available_dates = load_data()

# UI
st.title("📈🌎 MacroMind")

commodity = st.selectbox("Select a commodity", commodities)
startdate = st.selectbox("Select a start date", sorted(available_dates, reverse=True))
assistant_mode = st.selectbox("Assistant Mode", ["crisp","verbose"])

if st.button("Predict Direction"):
    with st.spinner("Predicting..."):
        prediction = com_pred(commodity, startdate,assistant_mode)
    st.success(f"📊 The predicted direction for **{commodity}** starting **{startdate}** is: **{prediction}**")
