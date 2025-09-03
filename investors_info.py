import streamlit as st
import pandas as pd
import requests
import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys from environment variables or secrets
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
if not FINNHUB_API_KEY and hasattr(st, 'secrets') and 'FINNHUB_API_KEY' in st.secrets:
    FINNHUB_API_KEY = st.secrets['FINNHUB_API_KEY']

# Common stock symbol corrections
SYMBOL_CORRECTIONS = {
    "AMAZON": "AMZN",
    "GOOGLE": "GOOGL",
    "META": "META",
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "NETFLIX": "NFLX",
    "TESLA": "TSLA"
}

def get_finnhub_data(ticker: str) -> Optional[Dict]:
    """Get stock data from Finnhub API"""
    try:
        # Debug: Print API key (first few characters)
        st.write(f"Using API key: {FINNHUB_API_KEY[:5]}...")
        
        headers = {'X-Finnhub-Token': FINNHUB_API_KEY}
        base_url = "https://finnhub.io/api/v1"
        
        # Get quote data first (this is the most reliable endpoint)
        quote_url = f"{base_url}/quote?symbol={ticker}"
        quote_response = requests.get(quote_url, headers=headers, timeout=10)
        
        if quote_response.status_code != 200:
            st.error(f"Error accessing Finnhub API: {quote_response.status_code}")
            st.write("Response:", quote_response.text)
            return None
            
        quote_data = quote_response.json()
        
        # If we can't get basic quote data, the ticker might be invalid
        if not quote_data or quote_data.get('c') is None:
            st.error("Could not get basic stock data")
            return None
            
        # Get company profile
        profile_url = f"{base_url}/stock/profile2?symbol={ticker}"
        profile_response = requests.get(profile_url, headers=headers, timeout=10)
        profile_data = profile_response.json() if profile_response.status_code == 200 else {}
        
        # Get basic financials
        metrics_url = f"{base_url}/stock/metric?symbol={ticker}&metric=all"
        metrics_response = requests.get(metrics_url, headers=headers, timeout=10)
        metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
        
        # Get company peers
        peers_url = f"{base_url}/stock/peers?symbol={ticker}"
        peers_response = requests.get(peers_url, headers=headers, timeout=10)
        peers_data = peers_response.json() if peers_response.status_code == 200 else []
        
        # Current price and market cap
        current_price = quote_data.get('c', 0)
        market_cap = current_price * (profile_data.get('shareOutstanding', 0) or 0)
        
        result = {
            'name': profile_data.get('name', ticker),
            'current_price': current_price,
            'change_percent': quote_data.get('dp', 0),
            'high_day': quote_data.get('h', 0),
            'low_day': quote_data.get('l', 0),
            'sector': profile_data.get('finnhubIndustry', 'N/A'),
            'industry': profile_data.get('finnhubIndustry', 'N/A'),
            'market_cap': market_cap,
            'currency': profile_data.get('currency', 'USD'),
            'country': profile_data.get('country', 'N/A'),
            'exchange': profile_data.get('exchange', 'N/A'),
            'peers': peers_data if isinstance(peers_data, list) else []
        }
        
        # Add metrics if available
        if isinstance(metrics_data.get('metric'), dict):
            metrics = metrics_data['metric']
            result.update({
                'pe_ratio': metrics.get('peBasicExcl', 'N/A'),
                'dividend_yield': metrics.get('dividendYieldIndicatedAnnual', 'N/A'),
                'beta': metrics.get('beta', 'N/A')
            })
        
        return result
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def format_market_cap(market_cap: float) -> str:
    """Format market cap value to human readable string"""
    if market_cap == 0 or market_cap == 'N/A':
        return 'N/A'
    
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.2f}M"
    else:
        return f"${market_cap:,.0f}"

def validate_ticker(ticker: str) -> str:
    """Validate and correct common ticker symbols"""
    ticker = ticker.upper().strip()
    return SYMBOL_CORRECTIONS.get(ticker, ticker)

def show_investors():
    st.title("📊 Stock Market Analysis")
    
    # Input for stock ticker
    st.write("Enter the stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft)")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_ticker = st.text_input("Enter the Stock Ticker:", key="stock_input").upper().strip()
    
    with col2:
        if st.button("🔄 Refresh", key="refresh_button"):
            st.cache_data.clear()
            st.rerun()
    
    if stock_ticker:
        stock_ticker = validate_ticker(stock_ticker)
        
        with st.spinner(f"Loading data for {stock_ticker}..."):
            data = get_finnhub_data(stock_ticker)
            
            if data:
                st.subheader(f"📈 {data['name']} ({stock_ticker})")
                
                # Current Price and Change
                price_col1, price_col2 = st.columns(2)
                with price_col1:
                    st.metric("Current Price", f"${data['current_price']:.2f}", 
                             f"{data['change_percent']:.2f}%")
                with price_col2:
                    st.metric("Day Range", f"${data['low_day']:.2f} - ${data['high_day']:.2f}")
                
                # Company Overview
                st.subheader("🏢 Company Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if data['sector'] != 'N/A':
                        st.write(f"**Sector:** {data['sector']}")
                    if data['country'] != 'N/A':
                        st.write(f"**Country:** {data['country']}")
                with col2:
                    if data['industry'] != 'N/A':
                        st.write(f"**Industry:** {data['industry']}")
                    if data['exchange'] != 'N/A':
                        st.write(f"**Exchange:** {data['exchange']}")
                with col3:
                    st.write(f"**Market Cap:** {format_market_cap(data['market_cap'])}")
                    st.write(f"**Currency:** {data['currency']}")
                
                # Key Metrics
                if 'pe_ratio' in data:
                    st.subheader("📊 Key Metrics")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    with metrics_col1:
                        st.metric("P/E Ratio", f"{data['pe_ratio']}" if data['pe_ratio'] != 'N/A' else 'N/A')
                    with metrics_col2:
                        st.metric("Dividend Yield", f"{data['dividend_yield']}%" if data['dividend_yield'] != 'N/A' else 'N/A')
                    with metrics_col3:
                        st.metric("Beta", f"{data['beta']}" if data['beta'] != 'N/A' else 'N/A')
                
                # Peer Companies
                if data.get('peers'):
                    st.subheader("🤝 Peer Companies")
                    peers = [peer for peer in data['peers'] if peer != stock_ticker]
                    if peers:
                        st.write(" | ".join([f"`{peer}`" for peer in peers[:10]]))
                    else:
                        st.info("No peer companies data available")
            else:
                st.error(f"Could not fetch data for {stock_ticker}. Please verify the ticker symbol and try again.")
                st.info("Try these popular tickers: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon)")

if __name__ == "__main__":
    show_investors()

