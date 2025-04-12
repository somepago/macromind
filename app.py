import streamlit as st
from main import CommodityPricePredictor
import pandas as pd
from constants import current_news_data, current_stock_data

# Set config first!
st.set_page_config(page_title="MacroMind", layout="centered")

# Load once
@st.cache_data
def load_data():
    price_df = pd.read_csv(current_stock_data)
    news_path = current_news_data
    com_pred = CommodityPricePredictor(news_df=news_path, price_df="data_prep/commodity_data_60days.csv")
    available_commodities = sorted(price_df['Commodity'].unique().tolist())
    dates = pd.to_datetime(price_df['Date']).dt.strftime('%Y-%m-%d').unique().tolist()
    return com_pred, available_commodities, dates

com_pred, commodities, available_dates = load_data()

# UI
st.title("📈🌎 MacroMind")
st.markdown("Get market movement forecasts across multiple commodities based on current events and price history.")

# Selection
startdate = st.selectbox("📅 Select a start date", sorted(available_dates, reverse=True))
assistant_mode = st.selectbox("🧠 Assistant Mode", ["crisp", "verbose"])

# Predict All
if st.button("🔮 Predict Direction for All Commodities"):
    with st.spinner("Running predictions..."):
        predictions, explanations = com_pred.pred_all(commodities, startdate, assistant_mode)

    st.markdown("## 📊 Predictions for All Commodities")

    for idx, commodity in enumerate(commodities):
        pred = predictions[idx].upper()
        explanation = explanations[idx]

        # Choose tag color
        if pred == "UP":
            tag_color = "#d4edda"
            text_color = "#155724"
        elif pred == "DOWN":
            tag_color = "#fff3cd"
            text_color = "#856404"
        else:
            tag_color = "#d1ecf1"
            text_color = "#0c5460"

        # Display each prediction in a styled container
        with st.container():
            st.markdown(f"""
                <div style='border: 1px solid #ccc; border-radius: 10px; padding: 10px 15px; margin-bottom: 10px; background-color: #fdfdfd;'>
                    <h5 style='margin-bottom: 5px;'>📌 {commodity.title()}</h5>
                    <span style='background-color:{tag_color}; color:{text_color}; padding: 6px 14px; border-radius: 6px; font-weight: 600; font-size: 16px;'>
                        {pred}
                    </span>
                </div>
            """, unsafe_allow_html=True)

            # Show explanation button
            with st.expander("🔍 Explain Prediction"):
                st.markdown(explanation)
