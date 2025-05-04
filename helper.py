# helper.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import pipeline
from keybert import KeyBERT
from wordcloud import WordCloud
from statsmodels.tsa.arima.model import ARIMA as ARIMA_model # Renamed to avoid conflict
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the News API key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Base URL for News API
BASE_URL = "https://newsapi.org/v2/everything"

# --- 1. Data Collection ---

def fetch_crypto_data(ticker, start_date, end_date):
    """Fetches crypto OHLC data using yfinance."""
    try:
        crypto_data = yf.download(ticker, start=start_date, end=end_date)
        # Ensure columns are flat if MultiIndex was created
        if isinstance(crypto_data.columns, pd.MultiIndex):
             crypto_data.columns = crypto_data.columns.get_level_values(0)
        return crypto_data
    except Exception as e:
        print(f"Error fetching crypto data for {ticker}: {e}")
        return pd.DataFrame() # Return empty dataframe on error

def fetch_news_data(query="crypto"):
    """Collects news data using News API."""
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "language": "en" # Added language parameter
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        news_data = response.json()
        news_articles = news_data.get('articles', [])
        news_df = pd.DataFrame(news_articles)
        # Basic cleaning
        news_df = news_df.dropna(subset=['title', 'content'])
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
        return news_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news data: {e}")
        return pd.DataFrame() # Return empty dataframe on error
    except Exception as e:
        print(f"An unexpected error occurred while fetching news: {e}")
        return pd.DataFrame()

# --- 2. Data Preprocessing & Feature Engineering ---

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean() # Use EWM for smoother RSI
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    # Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_technical_indicators(crypto_data):
    """Adds SMA, EMA, and RSI to the crypto data."""
    data = crypto_data.copy()
    data['SMA_20'] = calculate_sma(data, 20)
    data['EMA_20'] = calculate_ema(data, 20)
    data['RSI'] = calculate_rsi(data)
    data.dropna(inplace=True)
    return data

# --- 3. Sentiment Analysis ---

# Using a generic sentiment model pipeline
# For better results, fine-tune or use a finance-specific model like FinBERT
@tf.keras.utils.register_keras_serializable() # Needed for caching TF models with Streamlit
def get_sentiment_pipeline():
    """Initializes and returns the sentiment analysis pipeline."""
    # Using a smaller, potentially faster model for demonstration if ProsusAI/finbert is too slow/large
    # return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(headline, sentiment_model):
    """Analyzes sentiment of a single headline."""
    if not headline or not isinstance(headline, str):
        return 'neutral', 0.5 # Default for empty/invalid headlines
    try:
        result = sentiment_model(headline)[0]
        # FinBERT often uses 'positive', 'negative', 'neutral'
        label_map = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
        return label_map.get(result['label'].lower(), 'neutral'), result['score']
    except Exception as e:
        print(f"Error analyzing sentiment for '{headline}': {e}")
        return 'neutral', 0.5 # Default on error

def perform_sentiment_analysis(news_df, sentiment_model):
    """Applies sentiment analysis to the news DataFrame."""
    if news_df.empty or 'title' not in news_df.columns:
        return news_df

    sentiments = news_df['title'].apply(lambda x: analyze_sentiment(x, sentiment_model))
    news_df[['sentiment', 'sentiment_score']] = pd.DataFrame(sentiments.tolist(), index=news_df.index)
    return news_df

# --- Keyword Extraction ---
@tf.keras.utils.register_keras_serializable() # Needed for caching models
def get_keybert_model():
    """Initializes and returns the KeyBERT model."""
    # Using a common, effective model for KeyBERT
    return KeyBERT(model='all-MiniLM-L6-v2') # Or 'distilbert-base-nli-mean-tokens'

def extract_keywords(text, kw_model, num_keywords=5):
    """Extracts keywords from text using KeyBERT."""
    if not text or not isinstance(text, str):
        return []
    try:
        # Adjust top_n dynamically or keep fixed
        # n_keywords = min(num_keywords, (len(text.split()) // 10) + 1) # Example dynamic adjustment
        keywords = kw_model.extract_keywords(text,
                                            keyphrase_ngram_range=(1, 1),
                                            stop_words='english',
                                            top_n=num_keywords) # Use fixed number
        return [kw[0] for kw in keywords if kw] # Ensure keyword exists
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def add_keywords_to_news(news_df, kw_model):
    """Adds a 'Keywords' column to the news DataFrame."""
    if news_df.empty or 'content' not in news_df.columns:
        news_df['Keywords'] = [[] for _ in range(len(news_df))]
        return news_df

    keywords_column = []
    for text in news_df['content']:
        keywords = extract_keywords(text, kw_model, num_keywords=10) # Extract top 5 keywords
        keywords_column.append(keywords)

    news_df['Keywords'] = keywords_column
    return news_df

def generate_wordcloud_figure(news_df):
    """Generates a WordCloud figure from keywords."""
    if 'Keywords' not in news_df.columns or news_df['Keywords'].isnull().all():
        # Return an empty figure or a placeholder message
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No keywords found to generate word cloud.",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

    all_keywords_list = [kw for sublist in news_df['Keywords'].dropna() for kw in sublist]
    if not all_keywords_list:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No keywords available.",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

    all_keywords_text = ' '.join(all_keywords_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


# --- 4. Model Building and Prediction ---

# 4.1 ARIMA Model
def train_arima_model(train_data_series, order=(5, 1, 0)):
    """Trains an ARIMA model."""
    try:
        model = ARIMA_model(train_data_series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        return None

def forecast_arima(model_fit, steps):
    """Generates forecasts using a fitted ARIMA model."""
    try:
        return model_fit.forecast(steps=steps)
    except Exception as e:
        print(f"Error forecasting with ARIMA: {e}")
        return None

# 4.2 LSTM Model
def prepare_lstm_data(data, target_column='Close', seq_length=60):
    """Prepares data for LSTM: scaling and sequence creation."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Ensure we only scale numeric data and the target column exists
    numeric_cols = data.select_dtypes(include=np.number).columns
    if target_column not in numeric_cols:
        raise ValueError(f"Target column '{target_column}' not found or not numeric.")

    scaled_data = scaler.fit_transform(data[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=data.index)

    # Find index of target column for sequence creation
    target_col_idx = list(numeric_cols).index(target_column)

    xs, ys = [], []
    for i in range(len(scaled_data) - seq_length):
        xs.append(scaled_data[i : i + seq_length])
        ys.append(scaled_data[i + seq_length, target_col_idx]) # Predict only the target column
    return np.array(xs), np.array(ys), scaler, scaled_df.columns # Return columns used for scaling

def build_lstm_model(seq_length, input_dim):
    """Builds the LSTM model architecture."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, input_dim)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1) # Predicts 1 value (the target column)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Note: Training is computationally intensive. Caching the *trained* model is key.
def train_lstm_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Trains the LSTM model."""
    # Consider adding validation split or early stopping for better training
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)
    return model, history

def predict_lstm(model, X_test, scaler, scaled_columns, target_column='Close'):
    """Generates predictions using the LSTM model and inverse transforms."""
    predictions_scaled = model.predict(X_test)

    # We need to inverse transform correctly.
    # Create a dummy array with the same shape as the original scaled data
    # Put the predictions into the correct column index before inverse transforming.
    target_col_idx = list(scaled_columns).index(target_column)
    dummy_array = np.zeros((predictions_scaled.shape[0], len(scaled_columns)))
    dummy_array[:, target_col_idx] = predictions_scaled.flatten()

    # Inverse transform using the full scaler
    predictions = scaler.inverse_transform(dummy_array)[:, target_col_idx]
    return predictions.flatten() # Return as a flat array


# 4.3 Random Forest Model (Example - Needs Adaptation for Prediction Task)
# This setup predicts the current close based on past indicators.
# For forecasting future close, you'd shift the target.
def prepare_rf_data(data_with_indicators):
    """Prepares features and target for Random Forest."""
    # Predict next day's close: shift target by -1
    data_with_indicators['Target'] = data_with_indicators['Close'].shift(-1)
    data_with_indicators.dropna(inplace=True) # Drop last row with NaN target

    features = data_with_indicators[['Close', 'SMA_20', 'EMA_20', 'RSI', 'Open', 'High', 'Low', 'Volume']]
    target = data_with_indicators['Target']
    return features, target

def train_rf_model(X_train, y_train, n_estimators=100, random_state=42):
    """Trains a Random Forest Regressor."""
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1) # Use all CPU cores
    rf_model.fit(X_train, y_train)
    return rf_model

def predict_rf(model, X_test):
    """Generates predictions using the Random Forest model."""
    return model.predict(X_test)




from groq import Client
client = Client(api_key=GROQ_API_KEY)

# --------------CHATBOT API-----------------------
def generate_response(query):
    """Generate chatbot response using RAG."""

    prompt = f"Based on the crypto news context below, answer the query as an expert:\n\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content


# --- 5. Visualization ---

def plot_crypto_timeseries(data, ticker):
    """Plots the closing price time-series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title=f'{ticker} Closing Price Over Time',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      hovermode='x unified')
    return fig

def plot_technical_indicators(data, ticker):
    """Plots Close Price with SMA and EMA."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', opacity=0.7))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(dash='dot')))
    fig.update_layout(title=f'{ticker}: Close Price with Moving Averages',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)',
                      hovermode='x unified')
    return fig

def plot_rsi(data, ticker):
    """Plots the RSI."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='orange')))
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="bottom right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")
    fig.update_layout(title=f'{ticker}: Relative Strength Index (RSI)',
                      xaxis_title='Date',
                      yaxis_title='RSI Value (0-100)',
                      yaxis_range=[0,100],
                       hovermode='x unified')
    return fig

def plot_sentiment_distribution(news_df):
    """Plots the distribution of news sentiment labels."""
    if news_df.empty or 'sentiment' not in news_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No sentiment data available.", ha='center', va='center')
        ax.axis('off')
        return fig

    plt.style.use('seaborn-v0_8-darkgrid') # Use a seaborn style
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='sentiment', data=news_df, ax=ax, palette='viridis', order=news_df['sentiment'].value_counts().index)
    ax.set_title('Sentiment Distribution in Crypto News Headlines')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Articles')
    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    return fig

def plot_sentiment_scores(news_df):
    """Plots the distribution of news sentiment scores."""
    if news_df.empty or 'sentiment_score' not in news_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No sentiment score data available.", ha='center', va='center')
        ax.axis('off')
        return fig

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(news_df['sentiment_score'], kde=True, ax=ax, color='purple', bins=10)
    ax.set_title('Sentiment Score Distribution')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig

def plot_candlestick_comparison(actual_data, predicted_close, title, predicted_col_name='Predicted_Close'):
    """Plots actual candlestick chart overlaid with predicted close."""
    fig = go.Figure()

    # Actual Candlestick
    fig.add_trace(go.Candlestick(
        x=actual_data.index,
        open=actual_data['Open'],
        high=actual_data['High'],
        low=actual_data['Low'],
        close=actual_data['Close'],
        name='Actual OHLC'
    ))

    # Predicted Close Line (or Candlestick if Open/High/Low predicted)
    # Assuming only Close is predicted
    if predicted_close is not None and len(predicted_close) == len(actual_data):
         # Ensure predicted_close is a Series with the same index
        if isinstance(predicted_close, (np.ndarray, list)):
             predicted_close = pd.Series(predicted_close, index=actual_data.index)

        fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=predicted_close,
            mode='lines',
            name=predicted_col_name,
            line=dict(color='cyan', width=2, dash='dot')
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False, # Hide rangeslider for clarity
        hovermode='x unified'
    )
    return fig

def plot_candlestick_prediction_ohlc(test_data_orig, predicted_data_ohlc, title="Comparison: Actual vs Predicted Candlestick"):
    """ Creates a Plotly figure comparing actual and predicted OHLC candlesticks."""
    test_data = test_data_orig.copy() # Avoid modifying original data

    # Ensure predicted data has the same index
    predicted_data_ohlc.index = test_data.index

    fig = go.Figure()

    # Actual Candlestick
    fig.add_trace(go.Candlestick(
        x=test_data.index,
        open=test_data['Open'],
        high=test_data['High'],
        low=test_data['Low'],
        close=test_data['Close'],
        name='Actual'
    ))

    # Predicted Candlestick
    fig.add_trace(go.Candlestick(
        x=predicted_data_ohlc.index,
        open=predicted_data_ohlc['Open'],
        high=predicted_data_ohlc['High'],
        low=predicted_data_ohlc['Low'],
        close=predicted_data_ohlc['Close'],
        name='Predicted',
        increasing_line_color='cyan',
        decreasing_line_color='magenta',
        opacity=0.7
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False # Optional: hide the range slider
    )
    return fig


# --- ARIMA Specific Prediction & Plotting ---
def create_predicted_ohlc_arima(test_data, predicted_close):
    """Creates a DataFrame with predicted OHLC based on ARIMA close forecast."""
    if predicted_close is None:
        return pd.DataFrame()

    predicted_data = test_data.copy()
    # Make sure indices match before assigning
    if len(predicted_close) == len(predicted_data):
         predicted_data.index = predicted_close.index # Align indices if needed (ARIMA forecast might have different index)
         predicted_data['Close'] = predicted_close.values
    else:
        print(f"Warning: Length mismatch between test data ({len(predicted_data)}) and ARIMA prediction ({len(predicted_close)}). Cannot align.")
        # Handle mismatch: e.g., slice or return empty
        return pd.DataFrame()


    # Heuristic for predicted High/Low based on predicted Close and previous Open
    # This is a simple approximation; real prediction would be more complex
    predicted_data['High'] = predicted_data[['Open', 'Close']].max(axis=1) * (1 + 0.01) # Small margin
    predicted_data['Low'] = predicted_data[['Open', 'Close']].min(axis=1) * (1 - 0.01)  # Small margin

    # Ensure High is always >= max(Open, Close) and Low <= min(Open, Close)
    predicted_data['High'] = predicted_data[['High', 'Open', 'Close']].max(axis=1)
    predicted_data['Low'] = predicted_data[['Low', 'Open', 'Close']].min(axis=1)

    return predicted_data[['Open', 'High', 'Low', 'Close']]


# --- LSTM Specific Prediction & Plotting ---
# LSTM model in the example predicts only Close price.
# The original code had a version predicting all OHLC, but let's stick to predicting Close
# for consistency unless explicitly asked to predict all four.

def create_predicted_ohlc_lstm(test_data_actual, predicted_close_lstm):
    """Creates a DataFrame with predicted OHLC based on LSTM close forecast."""
    if predicted_close_lstm is None or len(predicted_close_lstm) != len(test_data_actual):
        print(f"Warning: Length mismatch or None prediction for LSTM. Actual: {len(test_data_actual)}, Predicted: {len(predicted_close_lstm) if predicted_close_lstm is not None else 'None'}")
        return pd.DataFrame()

    predicted_data = test_data_actual.copy()
    predicted_data['Close'] = predicted_close_lstm # Assign predicted close

    # Use the same heuristic as ARIMA for High/Low prediction
    predicted_data['High'] = predicted_data[['Open', 'Close']].max(axis=1) * (1 + 0.01) # Small margin
    predicted_data['Low'] = predicted_data[['Open', 'Close']].min(axis=1) * (1 - 0.01)  # Small margin

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    predicted_data['High'] = predicted_data[['High', 'Open', 'Close']].max(axis=1)
    predicted_data['Low'] = predicted_data[['Low', 'Open', 'Close']].min(axis=1)

    return predicted_data[['Open', 'High', 'Low', 'Close']]