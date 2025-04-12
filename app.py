import streamlit as st
from predictor import CommodityPricePredictor
import pandas as pd
from constants import current_news_data, current_stock_data

# ---------- CONFIG ---------- #
st.set_page_config(
    page_title="MacroMind",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- STYLE INJECTION ---------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #e0f2fe, #f8fafc);
        background-attachment: fixed;
    }

    .header {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 20px;
    }

    .title-text {
        font-size: 3rem;
        font-weight: 700;
        color: #0f172a;
    }

    .sub-text {
        font-size: 1.2rem;
        color: #64748b;
    }

    .pred-box {
        background: rgba(255,255,255,0.7);
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease-in-out;
        width: 200px;  /* Set a fixed width for better alignment */
        margin-bottom: 20px;  /* Add space for explanation */
    }

    .pred-box:hover {
        transform: scale(1.01);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }

    .explanation-panel {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
    }

    .button-custom {
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        background: linear-gradient(90deg, #6366f1, #3b82f6);
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }

    .button-custom:hover {
        filter: brightness(1.1);
    }

    .commodity-container {
        display: grid;
        grid-template-columns: repeat(5, 1fr); /* 5 items per row */
        gap: 20px;
        margin-top: 20px;
    }

    .commodity-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- DATA LOAD ---------- #
@st.cache_data
def load_data():
    price_df = pd.read_csv(current_stock_data)
    news_path = current_news_data
    com_pred = CommodityPricePredictor(news_df=news_path, price_df="data_prep/commodity_data_60days.csv")
    available_commodities = sorted(price_df['Commodity'].unique().tolist())
    dates = pd.to_datetime(price_df['Date']).dt.strftime('%Y-%m-%d').unique().tolist()
    return com_pred, available_commodities, dates

com_pred, commodities, available_dates = load_data()

# ---------- HEADER ---------- #
st.markdown("""
    <div class="header">
        <div class='title-text'>MacroMind</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='sub-text'>Market sentiment meets machine learning. Explore, predict, and explain with style.</div><br>", unsafe_allow_html=True)

# ---------- CONTROLS ---------- #
col1, _ = st.columns([2, 1])
with col1:
    startdate = st.selectbox("Select Start Date", sorted(available_dates, reverse=True))

# ---------- SESSION STATE ---------- #
if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.explanations = None
    st.session_state.show_explanations = {}

if st.button("Predict Direction for All Commodities", use_container_width=True):
    with st.spinner("Crunching data and running models..."):
        preds, expls = com_pred.pred_all(commodities, startdate)
        st.session_state.predictions = preds
        st.session_state.explanations = expls
        st.session_state.show_explanations = {}

# ---------- FORECAST RESULTS ---------- #
st.markdown("<h2>✨ Forecast Results</h2>", unsafe_allow_html=True)

# Add CSS for styling
st.markdown("""
    <style>
    .up-tag, .down-tag {
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        margin: 10px 0;
        text-align: center;
        display: inline-block;
    }
    
    .up-tag {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .down-tag {
        background-color: #fef9c3;
        color: #854d0e;
    }

    .explanation-panel {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        grid-column: 1 / -1;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- DISPLAY RESULTS ---------- #
if st.session_state.predictions:
    st.markdown("### Forecast Results")

    # Create rows with 4 columns each
    for i in range(0, len(commodities), 4):
        row_commodities = commodities[i:i+4]
        cols = st.columns(4)
        row_explanations = {}

        # Display commodities in the current row
        for j, (col, commodity) in enumerate(zip(cols, row_commodities)):
            with col:
                pred = st.session_state.predictions[i+j].upper()
                explanation = st.session_state.explanations[i+j]
                key = f"exp_{commodity}"

                # Determine colors based on prediction
                bg_class = "up-tag" if pred == "UP" else "down-tag"

                # Display prediction box
                st.markdown(f"""
                <div class='pred-box'>
                    <h3>{commodity.title()}</h3>
                    <div class='{bg_class}'>{pred}</div>
                </div>
                """, unsafe_allow_html=True)

                # Toggle explanation button
                if st.button('Toggle Explanation', key=key):
                    st.session_state.show_explanations[key] = not st.session_state.show_explanations.get(key, False)
                
                # Store explanation if toggle is on
                if st.session_state.show_explanations.get(key):
                    row_explanations[commodity] = explanation
        
        # Display explanations for the current row below the commodities
        if row_explanations:
            st.markdown("<div class='explanation-panel'>")
            for commodity, explanation in row_explanations.items():
                st.markdown(f"**{commodity}**: {explanation}")
            st.markdown("</div>")


