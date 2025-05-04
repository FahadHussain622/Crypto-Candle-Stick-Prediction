import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import helper

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions ---
# Cache data fetching to avoid re-downloading on every interaction
@st.cache_data(ttl=600) # Cache for 10 minutes
def load_crypto_data(ticker, start_date, end_date):
    return helper.fetch_crypto_data(ticker, start_date, end_date)

@st.cache_data(ttl=1800) # Cache news for 30 minutes
def load_news_data(query="crypto"):
    return helper.fetch_news_data(query)

# Cache resource-intensive model loading/creation
@st.cache_resource
def load_sentiment_pipeline():
    # st.write("Loading sentiment analysis model...") # Show message
    return helper.get_sentiment_pipeline()

@st.cache_resource
def load_keybert_model():
    # st.write("Loading keyword extraction model...") # Show message
    return helper.get_keybert_model()

# Cache the result of sentiment analysis and keyword extraction
@st.cache_data(ttl=1800) # Cache processed news for 30 mins
def process_news_data(_news_df, _sentiment_pipeline, _kw_model):
    if _news_df.empty:
        return _news_df
    start_time = time.time()
    # st.write("Performing sentiment analysis...")
    news_with_sentiment = helper.perform_sentiment_analysis(_news_df.copy(), _sentiment_pipeline) # Use copy
    st.write(f"Sentiment analysis done in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    # st.write("Extracting keywords...")
    news_processed = helper.add_keywords_to_news(news_with_sentiment, _kw_model)
    # st.write(f"Keyword extraction done in {time.time() - start_time:.2f} seconds.")
    return news_processed

# Cache ARIMA model training and prediction
@st.cache_data(ttl=3600) # Cache model results for 1 hour
def get_arima_predictions(_crypto_data_processed, forecast_steps=30):
    if _crypto_data_processed.empty or len(_crypto_data_processed) < 50: # Need enough data
         st.warning("Not enough historical data for ARIMA modeling.")
         return None, None

    train_data = _crypto_data_processed['Close'][:-forecast_steps]
    test_data_actual = _crypto_data_processed.iloc[-forecast_steps:]

    if len(train_data) < 10: # Basic check for sufficient training data
        st.warning("Not enough training data points for ARIMA.")
        return None, test_data_actual

    st.write(f"Training ARIMA model on {len(train_data)} data points...")
    start_time = time.time()
    model_fit = helper.train_arima_model(train_data, order=(5, 1, 0)) # Example order
    st.write(f"ARIMA training done in {time.time() - start_time:.2f} seconds.")

    if model_fit:
        st.write(f"Forecasting next {forecast_steps} steps with ARIMA...")
        start_time = time.time()
        predicted_close_arima = helper.forecast_arima(model_fit, steps=forecast_steps)
        st.write(f"ARIMA forecasting done in {time.time() - start_time:.2f} seconds.")

        if predicted_close_arima is not None:
            predicted_ohlc_arima = helper.create_predicted_ohlc_arima(test_data_actual, predicted_close_arima)
            return predicted_ohlc_arima, test_data_actual
        else:
             st.error("ARIMA forecasting failed.")
             return None, test_data_actual
    else:
        st.error("ARIMA model training failed.")
        return None, test_data_actual

# Cache LSTM model training and prediction (VERY IMPORTANT for usability)
# Use cache_resource for the model object itself
@st.cache_resource # Cache the trained model object
def get_trained_lstm_model(_X_train, _y_train, seq_length, input_dim):
    st.write(f"Building & Training LSTM model (Sequence Length: {seq_length}, Input Dim: {input_dim})... This may take time.")
    start_time = time.time()
    model = helper.build_lstm_model(seq_length, input_dim)
    # Reduce epochs for faster demo in Streamlit, increase for better accuracy
    trained_model, history = helper.train_lstm_model(model, _X_train, _y_train, epochs=10, batch_size=32)
    st.write(f"LSTM training complete in {time.time() - start_time:.2f} seconds.")
    return trained_model

@st.cache_data(ttl=3600) # Cache the prediction results
def get_lstm_predictions(_crypto_data_processed, seq_length=60, forecast_horizon=0.2): # forecast_horizon as fraction
    if _crypto_data_processed.empty or len(_crypto_data_processed) < seq_length + 20: # Need enough data
        st.warning("Not enough historical data for LSTM modeling.")
        return None, None

    st.write("Preparing data for LSTM...")
    # Use relevant columns for LSTM (e.g., OHLCV)
    lstm_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_for_lstm = _crypto_data_processed[lstm_cols].copy() # Ensure we have these cols

    X, y, scaler, scaled_columns = helper.prepare_lstm_data(data_for_lstm, target_column='Close', seq_length=seq_length)
    input_dim = X.shape[2] # Number of features used (e.g., 5 for OHLCV)

    split_idx = int(len(X) * (1 - forecast_horizon))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test_scaled = X[split_idx:], y[split_idx:] # y_test is scaled

    test_data_actual = _crypto_data_processed.iloc[-len(X_test):] # Get corresponding original data

    if len(X_train) < 10 or len(X_test) < 1:
        st.warning("Not enough data points after splitting for LSTM training/testing.")
        return None, test_data_actual

    # --- Train or Load Cached Model ---
    # The key includes training data shapes to ensure cache validity
    trained_model = get_trained_lstm_model(X_train, y_train, seq_length, input_dim)
    # --- ---

    if trained_model:
        st.write("Generating LSTM predictions...")
        start_time = time.time()
        predicted_close_lstm = helper.predict_lstm(trained_model, X_test, scaler, scaled_columns, target_column='Close')
        st.write(f"LSTM prediction done in {time.time() - start_time:.2f} seconds.")

        if predicted_close_lstm is not None:
             predicted_ohlc_lstm = helper.create_predicted_ohlc_lstm(test_data_actual, predicted_close_lstm)
             return predicted_ohlc_lstm, test_data_actual
        else:
            st.error("LSTM prediction failed.")
            return None, test_data_actual
    else:
        st.error("LSTM model training/loading failed.")
        return None, test_data_actual


# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")
crypto_ticker = st.sidebar.text_input("Crypto Ticker", 'BTC-USD').upper()
# Date Inputs
today = datetime.date.today()
default_start = today - datetime.timedelta(days=3*365) # Default 3 years
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

# Validate dates
if start_date >= end_date:
    st.sidebar.error("Error: End date must fall after start date.")
    st.stop() # Stop execution if dates are invalid

st.sidebar.markdown("---")
st.sidebar.header("📊 Data Previews")

# Load data using cached functions
crypto_data = load_crypto_data(crypto_ticker, start_date, end_date)
news_data_raw = load_news_data(query=f"{crypto_ticker} OR cryptocurrency OR crypto OR bitcoin OR ethereum") # Broader query

if not crypto_data.empty:
    st.sidebar.subheader(f"{crypto_ticker} Data Preview")
    st.sidebar.dataframe(crypto_data.tail())
    st.sidebar.metric(label="Latest Close Price", value=f"${crypto_data['Close'].iloc[-1]:,.2f}")

    # Add indicators (do this after loading)
    crypto_data_processed = helper.add_technical_indicators(crypto_data)

else:
    st.sidebar.warning(f"Could not fetch data for {crypto_ticker}.")
    crypto_data_processed = pd.DataFrame() # Ensure it's an empty DF

if not news_data_raw.empty:
    st.sidebar.subheader("Latest News Preview")
    st.sidebar.dataframe(news_data_raw[['title', 'source', 'publishedAt']].head())
    # Load models needed for news processing
    sentiment_pipeline = load_sentiment_pipeline()
    keybert_model = load_keybert_model()
    # Process news data (sentiment + keywords) using caching
    news_data_processed = process_news_data(news_data_raw, sentiment_pipeline, keybert_model)

else:
    st.sidebar.warning("Could not fetch news data.")
    news_data_processed = pd.DataFrame() # Ensure it's an empty DF

st.sidebar.markdown("---")
st.sidebar.info("Dashboard displays data and analysis based on selected configurations.")


# --- Main Dashboard Area ---
st.title(f"₿ {crypto_ticker} Analysis Dashboard")
st.markdown(f"Displaying data from **{start_date}** to **{end_date}**.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Crypto Data & Plots",
    "📰 Real-time News",
    "😊 News Sentiment",
    "🤖 AI Chatbot",
    "🕯️ OHLC & Predictions"
])

# --- Tab 1: Crypto Data & Plots ---
with tab1:
    st.header(f"{crypto_ticker} Price Data and Technical Indicators")

    if not crypto_data_processed.empty:
        # Display raw data (optional, can be large)
        # with st.expander("Show Raw Data with Indicators"):
        #     st.dataframe(crypto_data_processed)

        # Plotting
        st.plotly_chart(helper.plot_crypto_timeseries(crypto_data_processed, crypto_ticker), use_container_width=True)
        st.plotly_chart(helper.plot_technical_indicators(crypto_data_processed, crypto_ticker), use_container_width=True)
        st.plotly_chart(helper.plot_rsi(crypto_data_processed, crypto_ticker), use_container_width=True)
    else:
        st.warning("No cryptocurrency data available to display.")

# --- Tab 2: Real-time News ---
with tab2:
    st.header("Latest Crypto News Articles")
    if not news_data_processed.empty:
         # Display formatted news
         for index, row in news_data_processed.iterrows():
             st.subheader(row['title'])
             col1, col2 = st.columns([1, 4])
             with col1:
                 if row['urlToImage']:
                      st.image(row['urlToImage'], width=150)
                 else:
                     st.caption("No image")
             with col2:
                 st.write(f"**Source:** {row['source']['name']}")
                 st.write(f"**Published:** {row['publishedAt'].strftime('%Y-%m-%d %H:%M')}")
                 st.write(row['description'] if row['description'] else "No description available.")
                 st.markdown(f"[Read full article]({row['url']})", unsafe_allow_html=True)
             st.markdown("---") # Separator
    else:
        st.warning("No news articles available to display.")

# --- Tab 3: News Sentiment Analysis ---
with tab3:
    st.subheader("📈 Sentiment Distribution")
    news_data = news_data_processed
    if "sentiment" in news_data.columns:
        sentiment_counts = news_data["sentiment"].value_counts()

        # 🔹 Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # 🔹 Pie Chart in the first column
        with col1:
            fig_pie = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title="News Sentiment Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            

        # 🔹 Bar Chart in the second column
        with col2:
            fig_bar = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, 
                             color=sentiment_counts.index, title="Sentiment Count", 
                             labels={"x": "Sentiment", "y": "Count"})
            st.plotly_chart(fig_bar, use_container_width=True)

        # 🔹 Word Cloud
        st.subheader("🔤 Word Cloud of News Titles")
        
        if "title" in news_data.columns:
            text = " ".join(news_data["title"].dropna())  # Combine all titles
        else:
            text = " ".join(news_data["description"].dropna())  # Fallback to description
        
        wordcloud = WordCloud(width=700, height=350, background_color="black", colormap="cool").generate(text)
        
        # Display word cloud using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Sentiment data not found in news dataset!")



# --- Tab 4: AI Chatbot ---
with tab4:
    st.header("📰 Crypto Expertbot")

    # Function to map user/assistant roles for Streamlit
    def role_to_streamlit(role):
        return "assistant" if role == "bot" else "user"

    user_input = st.chat_input("Ask a question about recent news...")

    chat_container = st.container()

    # --- User Input Box (Always at Bottom) ---

    if user_input:
        # Show user message in chat
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Generate bot response
        with st.spinner("Generating Response..."):
            response_text = helper.generate_response(user_input)  # Call your RAG function

        # Show bot response in chat
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response_text)

# --- Tab 5: OHLC & Predictions ---
with tab5:
    st.header("OHLC Candlestick Charts and Model Predictions")

    if not crypto_data_processed.empty:
        # Use a portion of the data for testing predictions (e.g., last 30 days)
        forecast_steps = st.slider("Select Forecast Horizon (Days)", min_value=10, max_value=100, value=30, key="forecast_slider")

        st.subheader("ARIMA Model Prediction")
        # Get ARIMA predictions (cached)
        predicted_ohlc_arima, test_data_arima = get_arima_predictions(crypto_data_processed, forecast_steps=forecast_steps)

        if predicted_ohlc_arima is not None and not predicted_ohlc_arima.empty:
            fig_arima_compare = helper.plot_candlestick_prediction_ohlc(test_data_arima, predicted_ohlc_arima, title=f"ARIMA: Actual vs Predicted Candlestick ({forecast_steps} days)")
            st.plotly_chart(fig_arima_compare, use_container_width=True)
            with st.expander("View ARIMA Prediction Values"):
                # Combine actual and predicted for comparison
                comparison_df = test_data_arima[['Open', 'High', 'Low', 'Close']].copy()
                comparison_df['Predicted_Close_ARIMA'] = predicted_ohlc_arima['Close']
                st.dataframe(comparison_df)
        else:
             st.warning("ARIMA predictions could not be generated for the selected period.")

        st.markdown("---")

        st.subheader("LSTM Model Prediction")
        lstm_seq_length = 60 # Keep consistent or make it configurable
        # Get LSTM predictions (cached)
        predicted_ohlc_lstm, test_data_lstm = get_lstm_predictions(crypto_data_processed, seq_length=lstm_seq_length, forecast_horizon=forecast_steps/len(crypto_data_processed)) # Pass horizon as fraction

        if predicted_ohlc_lstm is not None and not predicted_ohlc_lstm.empty:
            fig_lstm_compare = helper.plot_candlestick_prediction_ohlc(test_data_lstm, predicted_ohlc_lstm, title=f"LSTM: Actual vs Predicted Candlestick ({len(test_data_lstm)} days)")
            st.plotly_chart(fig_lstm_compare, use_container_width=True)
            with st.expander("View LSTM Prediction Values"):
                 # Combine actual and predicted for comparison
                comparison_df_lstm = test_data_lstm[['Open', 'High', 'Low', 'Close']].copy()
                comparison_df_lstm['Predicted_Close_LSTM'] = predicted_ohlc_lstm['Close']
                st.dataframe(comparison_df_lstm)
        else:
             st.warning("LSTM predictions could not be generated for the selected period.")


    else:
        st.warning("No cryptocurrency data available for prediction.")

# --- Footer ---
st.markdown("---")
st.caption("Crypto Analysis Dashboard | Data Sources: Yahoo Finance, NewsAPI | Models: ARIMA, LSTM (TensorFlow/Keras), Sentiment (HuggingFace Transformers), Keywords (KeyBERT)")