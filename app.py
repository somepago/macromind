import streamlit as st
from predictor import CommodityPricePredictor
import pandas as pd
from constants import current_news_data, current_stock_data
from allocator import commodity_allocator
import matplotlib.pyplot as plt

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

# ---------- SESSION STATE ---------- #
if "initialized" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.explanations = None
    st.session_state.active_commodity = None
    st.session_state.sentiment_scores = None
    st.session_state.selected_commodities = []  # Start with an empty list instead of "ALL"
    st.session_state.showing_all = False  # Default to false so no commodities are pre-selected
    st.session_state.initialized = True

# ---------- CONTROLS ---------- #
col1, col2 = st.columns([1, 2])

with col1:
    startdate = st.selectbox("📅 Select Start Date", sorted(available_dates, reverse=True))

with col2:
    # Determine options based on current state
    options = commodities  # Allow "ALL" option
    default = []  # No commodities pre-selected by default
    
    # Multi-select box for commodities
    selected = st.multiselect(
        "Select Commodities to Predict",
        options=options,
        default=default,
        key="commodity_selector"
    )
    
    # Handle selection logic
    if len(selected) == 0:
        st.session_state.showing_all = False
        st.session_state.selected_commodities = []  # Empty selection
    elif "ALL" in selected:
        st.session_state.showing_all = True
        st.session_state.selected_commodities = ["ALL"]
    else:
        st.session_state.showing_all = False
        st.session_state.selected_commodities = selected

# Get the actual commodities to predict
selected_commodities = commodities if "ALL" in st.session_state.selected_commodities else st.session_state.selected_commodities

# ---------- FORECAST SECTION ---------- #
st.markdown("### ✨ Forecast")

# Add CSS for styling
st.markdown("""
<style>
    .pred-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        text-align: center;
    }
    .pred-box h3 {
        margin: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .up-tag, .down-tag {
        display: inline-block;
        padding: 0.25rem 1rem;
        border-radius: 1rem;
        font-weight: bold;
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

# Predict button (only trigger predictions when clicked)
if st.button("🔮 Generate Predictions", use_container_width=True):
    if len(st.session_state.selected_commodities) == 0:
        st.warning("Please select at least one commodity before generating predictions.")
    else:
        with st.spinner("Crunching data and running models..."):
            preds, expls, sentiment_scores = com_pred.pred_all(st.session_state.selected_commodities, startdate)
            st.session_state.predictions = preds
            st.session_state.explanations = expls
            st.session_state.sentiment_scores = sentiment_scores
            st.session_state.active_commodity = None  # Reset active commodity

# Display results only if predictions exist
if st.session_state.predictions:
    # ---------- DISPLAY RESULTS ---------- #
    st.markdown("#### Results")
    for i in range(0, len(st.session_state.selected_commodities), 4):
        row_commodities = st.session_state.selected_commodities[i:i + 4]
        cols = st.columns(4)
        
        # Display commodities in the current row
        for j, (col, commodity) in enumerate(zip(cols, row_commodities)):
            with col:
                pred_idx = st.session_state.selected_commodities.index(commodity)
                pred = st.session_state.predictions[pred_idx].upper()
                explanation = st.session_state.explanations[pred_idx]
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
                    # If clicking the same commodity, hide its explanation
                    if st.session_state.active_commodity == commodity:
                        st.session_state.active_commodity = None
                    else:
                        # Show this commodity's explanation and hide any other
                        st.session_state.active_commodity = commodity
        
        # Display explanation if active commodity is in current row
        if st.session_state.active_commodity in row_commodities:
            active_idx = st.session_state.selected_commodities.index(st.session_state.active_commodity)
            active_explanation = st.session_state.explanations[active_idx]
            st.markdown(f"""
            <div class='explanation-panel'>
                <strong>{st.session_state.active_commodity}</strong>: {active_explanation}
            </div>
            """, unsafe_allow_html=True)
# ---------- PORTFOLIO ALLOCATION ---------- #
    if st.session_state.predictions is not None:
        st.markdown("<h2>💰 Portfolio Allocation</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            budget = st.number_input("Investment Budget ($)", min_value=100.0, max_value=1000000.0, value=1000.0, step=100.0)
        
        with col2:
            strategy = st.selectbox(
                "Investment Strategy", 
                ["aggressive", "conservative", "linear", "neutral_aware"],
                help="Aggressive: Higher weights for UP predictions. Conservative: Lower weights overall. Linear: Direct correlation with sentiment. Neutral_aware: Considers neutral predictions."
            )
        
        with col3:
            temp = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.5, step=0.1, 
                             help="Lower values create sharper differences between allocations. Higher values make allocations more even.")
        
        if st.button("Generate Portfolio Allocation", use_container_width=True):
            # Use the precomputed predictions and sentiments for selected commodities
            allocation_df = commodity_allocator(
                com_pred=com_pred,
                commodities=st.session_state.selected_commodities,  # Use selected commodities instead of all
                startdate=startdate,
                budget=budget,
                strategy=strategy,
                temp=temp,
                precomputed_predictions=st.session_state.predictions,
                precomputed_sentiments=st.session_state.sentiment_scores
            )
            
            # Display the allocation table
            st.dataframe(
                allocation_df,
                column_config={
                    "Commodity": st.column_config.TextColumn("Commodity"),
                    "Prediction": st.column_config.TextColumn("Prediction"),
                    "Sentiment Score": st.column_config.ProgressColumn("Sentiment", format="%.2f", min_value=0, max_value=1),
                    "Allocation Weight": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1),
                    "Dollar Allocation": st.column_config.NumberColumn("Allocation ($)", format="$%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Create a pie chart for the allocations
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                allocation_df["Dollar Allocation"], 
                labels=[f"{row['Commodity']} (${row['Dollar Allocation']:.2f})" for _, row in allocation_df.iterrows()],
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                explode=[0.05] * len(allocation_df)
            )
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)