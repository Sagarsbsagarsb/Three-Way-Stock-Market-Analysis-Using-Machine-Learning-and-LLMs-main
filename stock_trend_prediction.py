import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import json
from typing import Optional, Dict, Any, List, Tuple

# Thread-local storage for session objects
thread_local = threading.local()

# API Keys - Should be stored in environment variables or Streamlit secrets
ALPHA_VANTAGE_KEY = st.secrets["ALPHA_VANTAGE_KEY"] if "ALPHA_VANTAGE_KEY" in st.secrets else None
FINNHUB_KEY = st.secrets["FINNHUB_KEY"] if "FINNHUB_KEY" in st.secrets else None
TWELVE_DATA_KEY = st.secrets["TWELVE_DATA_KEY"] if "TWELVE_DATA_KEY" in st.secrets else None
POLYGON_KEY = st.secrets["POLYGON_KEY"] if "POLYGON_KEY" in st.secrets else None
TIINGO_KEY = st.secrets["TIINGO_KEY"] if "TIINGO_KEY" in st.secrets else None

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    return thread_local.session

def fetch_alpha_vantage_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch data from Alpha Vantage API"""
    if not ALPHA_VANTAGE_KEY:
        return None
    
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            return df
    except Exception as e:
        st.warning(f"Alpha Vantage API error: {str(e)}")
    return None

def fetch_polygon_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch data from Polygon.io API"""
    if not POLYGON_KEY:
        return None
    
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}?apiKey={POLYGON_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df.index = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
    except Exception as e:
        st.warning(f"Polygon.io API error: {str(e)}")
    return None

def fetch_tiingo_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch data from Tiingo API"""
    if not TIINGO_KEY:
        return None
    
    try:
        headers = {'Authorization': f'Token {TIINGO_KEY}'}
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_str}&endDate={end_str}"
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if data:
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df['date'])
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
    except Exception as e:
        st.warning(f"Tiingo API error: {str(e)}")
    return None

def fetch_yahoo_finance_data(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if not df.empty:
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.warning(f"Yahoo Finance error: {str(e)}")
    return None

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    try:
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Additional indicators
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['20dSTD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df
    except Exception as e:
        st.warning(f"Error adding technical indicators: {str(e)}")
        return df

def fetch_company_info(ticker: str) -> Dict[str, Any]:
    """Fetch company information from multiple sources"""
    info = {
        'company_name': ticker,
        'currency': 'USD',
        'sector': 'N/A',
        'industry': 'N/A'
    }
    
    # Try Yahoo Finance
    try:
        yf_stock = yf.Ticker(ticker)
        yf_info = yf_stock.info
        info.update({
            'company_name': yf_info.get('longName', ticker),
            'currency': yf_info.get('currency', 'USD'),
            'sector': yf_info.get('sector', 'N/A'),
            'industry': yf_info.get('industry', 'N/A')
        })
        return info
    except:
        pass
    
    # Try Alpha Vantage company overview
    if ALPHA_VANTAGE_KEY:
        try:
            url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
            response = requests.get(url, timeout=10)
            av_info = response.json()
            if av_info:
                info.update({
                    'company_name': av_info.get('Name', ticker),
                    'sector': av_info.get('Sector', 'N/A'),
                    'industry': av_info.get('Industry', 'N/A')
                })
        except:
            pass
    
    return info

# Enable caching for data fetching with increased TTL
@st.cache_data(ttl=6*3600)  # Cache for 6 hours
def fetch_stock_data(ticker: str, max_retries: int = 5) -> Optional[pd.DataFrame]:
    """Fetch stock data from multiple sources in parallel"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Define data sources with their fetching functions
    data_sources = [
        (fetch_alpha_vantage_data, "Alpha Vantage"),
        (fetch_polygon_data, "Polygon.io"),
        (fetch_tiingo_data, "Tiingo"),
        (fetch_yahoo_finance_data, "Yahoo Finance")
    ]
    
    results = {}
    with ThreadPoolExecutor(max_workers=len(data_sources)) as executor:
        future_to_source = {
            executor.submit(fetch_func, ticker, start_date, end_date): source_name
            for fetch_func, source_name in data_sources
        }
        
        # As each source completes, check if we have valid data
        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[source_name] = df
            except Exception as e:
                st.warning(f"Error with {source_name}: {str(e)}")
    
    # If we have results, use the source with the most data points
    if results:
        best_source = max(results.items(), key=lambda x: len(x[1]))
        st.success(f"Data fetched successfully from {best_source[0]}")
        
        # Add technical indicators
        df = add_technical_indicators(best_source[1])
        
        # Fetch and add company info
        company_info = fetch_company_info(ticker)
        df.attrs.update(company_info)
        
        return df
    
    st.error("Unable to fetch data from any source. Please try again later.")
    return None

# Batch fetch function for multiple stocks
def fetch_multiple_stocks(tickers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        results = list(executor.map(fetch_stock_data, tickers))
    return {ticker: result for ticker, result in zip(tickers, results) if result is not None}

# Predefined stock tickers (Popular Companies)
STOCK_TICKERS = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Google (Alphabet) (GOOGL)": "GOOGL",
    "Tesla Inc. (TSLA)": "TSLA",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "Netflix Inc. (NFLX)": "NFLX"
}

@st.cache_resource
def load_keras_model():
    return load_model('keras_model_3.h5')

def plot_technical_indicators(df, ticker):
    # Create figure with secondary y-axis
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot price and volume
    ax1.plot(df.index, df['Close'], 'b-', label='Price')
    ax2.bar(df.index, df['Volume'], alpha=0.3, color='gray', label='Volume')
    
    # Add moving averages
    ax1.plot(df.index, df['Close'].rolling(window=20).mean(), 'r--', label='20-day MA')
    ax1.plot(df.index, df['Close'].rolling(window=50).mean(), 'g--', label='50-day MA')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Price ({df.attrs.get("currency", "USD")})', color='b')
    ax2.set_ylabel('Volume', color='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'{df.attrs.get("company_name", ticker)} - {df.attrs.get("sector", "")}')
    plt.xticks(rotation=45)
    return fig

# Main function for stock trend prediction
def run_stock_trend_prediction():
    st.title('Stock Trend Prediction')

    # Dropdown for Ticker Selection
    selected_stock = st.selectbox("Select a Stock", list(STOCK_TICKERS.keys()))
    user_input = STOCK_TICKERS[selected_stock]

    with st.spinner('Fetching stock data...'):
        df = fetch_stock_data(user_input)

    if df is None or df.empty:
        st.error(f"Unable to fetch data for {selected_stock}.")
        st.error("Please try again or select a different stock.")
        return

    st.success("Data fetched successfully!")
    
    # Show company info
    st.subheader("Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Company:** {df.attrs.get('company_name', user_input)}")
        st.write(f"**Sector:** {df.attrs.get('sector', 'N/A')}")
    with col2:
        st.write(f"**Industry:** {df.attrs.get('industry', 'N/A')}")
        st.write(f"**Currency:** {df.attrs.get('currency', 'USD')}")
    
    # Show current stock info
    current_price = df['Close'].iloc[-1]
    price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
    volume = df['Volume'].iloc[-1]
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
    with col2:
        st.metric("Volume", f"{volume:,.0f}")
    with col3:
        st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

    # Technical Analysis Tab
    st.subheader("Technical Analysis")
    
    # Plot main chart with price and volume
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot price and volume
    ax1.plot(df.index, df['Close'], 'b-', label='Price')
    ax2.bar(df.index, df['Volume'], alpha=0.3, color='gray', label='Volume')
    
    # Add moving averages
    ax1.plot(df.index, df['MA20'], 'r--', label='20-day MA')
    ax1.plot(df.index, df['Upper_Band'], 'g--', label='Upper BB')
    ax1.plot(df.index, df['Lower_Band'], 'g--', label='Lower BB')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Price ({df.attrs.get("currency", "USD")})', color='b')
    ax2.set_ylabel('Volume', color='gray')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'{df.attrs.get("company_name", user_input)} - {df.attrs.get("sector", "")}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # MACD and RSI in separate charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### MACD")
        fig_macd = plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['MACD'], label='MACD')
        plt.plot(df.index, df['Signal_Line'], label='Signal Line')
        plt.title('MACD Analysis')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_macd)
    
    with col2:
        st.write("### RSI")
        fig_rsi = plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['RSI'], label='RSI')
        plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        plt.title('RSI Analysis')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_rsi)

    # Additional Technical Indicators
    st.subheader("Additional Technical Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Bollinger Bands")
        fig_bb = plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['Close'], label='Price')
        plt.plot(df.index, df['MA20'], label='20-day MA')
        plt.plot(df.index, df['Upper_Band'], label='Upper BB')
        plt.plot(df.index, df['Lower_Band'], label='Lower BB')
        plt.title('Bollinger Bands')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_bb)
    
    with col2:
        st.write("### Average True Range (ATR)")
        fig_atr = plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['ATR'], label='ATR')
        plt.title('Average True Range')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig_atr)

    # Price Prediction
    try:
        # Data preparation for prediction
        data = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Prepare sequences for prediction
        seq_length = 60
        x_test = []
        for i in range(seq_length, len(scaled_data)):
            x_test.append(scaled_data[i-seq_length:i])
        x_test = np.array(x_test)
        
        # Load model and make predictions
        model = load_keras_model()
        predictions = model.predict(x_test, verbose=0)
        
        # Scale predictions back
        predictions = scaler.inverse_transform(predictions)
        
        # Plot predictions
        st.subheader("Price Predictions")
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df.index[-len(predictions):], df['Close'][-len(predictions):], 'b', label='Actual Price')
        plt.plot(df.index[-len(predictions):], predictions, 'r', label='Predicted Price')
        plt.title('Price Predictions vs Actual')
        plt.xlabel('Date')
        plt.ylabel(f'Price ({df.attrs.get("currency", "USD")})')
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Show prediction metrics
        last_actual = df['Close'].iloc[-1]
        last_prediction = predictions[-1][0]
        prediction_change = ((last_prediction - last_actual) / last_actual) * 100
        
        st.write("### Prediction Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${last_actual:.2f}")
        with col2:
            st.metric("Predicted Next Price", f"${last_prediction:.2f}", f"{prediction_change:.2f}%")
            
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
