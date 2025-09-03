import streamlit as st

# Import existing pages
from stock_sentiment_analysis import run_stock_sentiment_analysis
from stock_trend_prediction import run_stock_trend_prediction
from investors_info import show_investors
from analyst_ratings import show_analyst_ratings

# Define the pages in the app
pages = {
    "Stock Sentiment Analysis": run_stock_sentiment_analysis,
    "Stock Trend Prediction": run_stock_trend_prediction,
    "Institutional & Major Investors": show_investors,
    "Analyst Ratings & Price Targets": show_analyst_ratings,
}

# Streamlit navigation
st.set_page_config(page_title="StockVision AI - Stock Market Dashboard", layout="wide")
st.title("📈 StockVision AI ")

page_selection = st.sidebar.selectbox("🔍 Select a page:", options=list(pages.keys()))

# Run the selected page
pages[page_selection]()
