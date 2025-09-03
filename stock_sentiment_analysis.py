import streamlit as st
from transformers import pipeline
import feedparser

# Initialize the sentiment analysis pipeline
pipe = pipeline(task="text-classification", model="ProsusAI/finbert")

def analyze_sentiment(ticker, keyword):
    rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    articles = []

    for entry in feed.entries:
        if keyword.lower() not in entry.summary.lower():
            continue

        sentiment = pipe(entry.summary)[0]
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published,
            'summary': entry.summary,
            'sentiment': sentiment['label'],
            'score': sentiment['score']
        })

        if sentiment['label'] == 'positive':
            total_score += sentiment['score']
            num_articles += 1
        elif sentiment['label'] == 'negative':
            total_score -= sentiment['score']
            num_articles += 1

    # Calculate final score and overall sentiment
    if num_articles > 0:
        final_score = total_score / num_articles
    else:
        final_score = 0

    if final_score > 0.2:
        overall_sentiment = "Positive"
    elif final_score < -0.2:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return overall_sentiment, final_score, articles

def run_stock_sentiment_analysis():
    st.title("Stock Sentiment Analysis")
    st.write("Enter the stock ticker and keyword to analyze recent news articles.")

    ticker_input = st.text_input("Stock Ticker (e.g., META):")
    keyword_input = st.text_input("Keyword (e.g., meta):")

    if st.button("Analyze"):
        if ticker_input and keyword_input:
            with st.spinner("Fetching and analyzing data..."):
                overall_sentiment, final_score, articles = analyze_sentiment(ticker_input, keyword_input)
            
            # Display results
            st.subheader("Overall Sentiment")
            st.write(f"Sentiment: {overall_sentiment}, Score: {final_score:.2f}")

            st.subheader("Articles")
            for article in articles:
                st.markdown(f"**Title:** [{article['title']}]({article['link']})")
                st.write(f"**Published:** {article['published']}")
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Sentiment:** {article['sentiment']}, Score: {article['score']:.2f}")
                st.write("---")
        else:
            st.error("Please enter both the stock ticker and keyword.")
