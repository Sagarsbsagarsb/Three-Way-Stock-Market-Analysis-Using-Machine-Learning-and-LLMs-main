import streamlit as st
import tweepy
from transformers import pipeline
import time
import re

# Replace with your Bearer Token
bearer_token = st.secrets["TWITTER_BEARER_TOKEN"] if "TWITTER_BEARER_TOKEN" in st.secrets else None

if not bearer_token:
    st.error("Please set up your Twitter Bearer Token in Streamlit secrets.")
    st.stop()

# Set up tweepy client using Bearer Token authentication
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline(task="text-classification", model="ProsusAI/finbert")

# Function to clean tweet text
def clean_tweet(text):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", '', text)
    text = re.sub(r"RT\s+", '', text)
    text = text.strip()
    return text

# Function to fetch tweets
def fetch_tweets(search_term, tweet_amount=10):
    try:
        # Add the 'lang:en' filter to the query to fetch only English tweets
        query = f"{search_term} lang:en"
        
        # Fetch tweets using the Tweepy Client with Bearer Token
        tweets = client.search_recent_tweets(query=query, max_results=tweet_amount)
        
        return tweets.data if tweets.data else []
    except tweepy.TweepyException as e:
        st.error(f"An error occurred: {e}")
        return []

# Function to analyze sentiment
def analyze_sentiment(tweets):
    results = []
    total_score = 0  # Initialize total score
    count = 0  # Initialize count of valid sentiments
    
    for tweet in tweets:
        cleaned_text = clean_tweet(tweet.text)
        sentiment = sentiment_pipeline(cleaned_text)[0]
        
        # Only include results with a confidence score above 0.7
        if sentiment['score'] > 0.7:
            results.append({
                'original_text': tweet.text,
                'cleaned_text': cleaned_text,
                'sentiment': sentiment['label'],
                'score': sentiment['score']
            })
            total_score += sentiment['score']  # Accumulate score
            count += 1  # Increment count of valid sentiments
        
        time.sleep(1)  # Added a delay to avoid hitting rate limits
    
    overall_sentiment = None
    overall_score = total_score / count if count > 0 else 0
    
    # Determine overall sentiment label based on average score
    if overall_score >= 0.75:
        overall_sentiment = "Positive"
    elif overall_score <= 0.25:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return results, overall_sentiment, overall_score

# Streamlit App
def run_twitter_sentiment_analysis():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a search term to analyze recent tweets.")

    search_term_input = st.text_input("Search Term (e.g., stocks):")
    tweet_amount_input = st.number_input("Number of Tweets to Fetch:", min_value=1, max_value=100, value=10)

    if st.button("Analyze"):
        if search_term_input:
            with st.spinner("Fetching and analyzing tweets..."):
                tweets = fetch_tweets(search_term_input, tweet_amount_input)
                if tweets:
                    analyzed_results, overall_sentiment, overall_score = analyze_sentiment(tweets)
                    
                    # Display results
                    if analyzed_results:
                        st.subheader("Sentiment Analysis Results")
                        for result in analyzed_results:
                            st.markdown(f"**Original Tweet:** {result['original_text']}")
                            st.markdown(f"**Cleaned Tweet:** {result['cleaned_text']}")
                            st.write(f"**Sentiment:** {result['sentiment']}, Score: {result['score']:.2f}")
                            st.write("---")
                        
                        # Display Overall Sentiment
                        st.subheader("Overall Sentiment")
                        st.write(f"**Overall Sentiment:** {overall_sentiment}, Average Score: {overall_score:.2f}")
                    else:
                        st.write("No tweets met the sentiment confidence threshold.")
                else:
                    st.write("No tweets found for the given search term.")
        else:
            st.error("Please enter a search term.")
# Run the app
if __name__ == "__main__":
    run_twitter_sentiment_analysis()
