# Candlestick Prediction and Sentiment-Driven Crypto Analytics

**Project:** Candlestick Prediction and Sentiment-Driven Crypto Analytics  
**Course:** Semester 6 – Data Mining Project  
**Date:** May 7, 2025  

**Project Members:**  
- Abdullah Maqsood  
- Ameer Tufail  
- Nouveen Leghari  
- Fahad Hussain  

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Objectives](#objectives)  
3. [Data Sources](#data-sources)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Sentiment Analysis](#sentiment-analysis)  
6. [Visualization and Dashboard](#visualization-and-dashboard)  
7. [Forecasting Models](#forecasting-models)  
8. [AI Chatbot Integration](#ai-chatbot-integration)  
9. [Conclusion](#conclusion)  
10. [Future Work](#future-work)  
11. [References](#references)  

---

## Introduction
Cryptocurrency markets are highly volatile and non-linear. This project predicts **OHLC (Open, High, Low, Close) candlestick values** using a combination of real-time market data, news sentiment analysis, and AI-based time series forecasting. The final product is an **interactive Streamlit dashboard** that provides traders and researchers with actionable insights.

---

## Objectives
- Fetch and visualize real-time OHLC data for major cryptocurrencies (BTC, ETH, etc.).  
- Analyze market sentiment using news headlines and NLP.  
- Predict candlestick values using ARIMA, LSTM, and Prophet models.  
- Evaluate model performance using standard metrics (MAE, MSE, RMSE, R²).  
- Provide an interactive user interface via Streamlit.

---

## Data Sources

### Cryptocurrency Data (OHLC)
- Collected using the **yfinance** library.  
- Includes historical and near real-time OHLC data for major cryptocurrencies.

### News Data
- Real-time crypto news obtained via **NewsAPI**.  
- Headlines, descriptions, and publication dates used for sentiment analysis.

---

## Data Preprocessing

### OHLC Data
- Parsing and formatting timestamps  
- Filling or dropping missing values  
- Scaling numeric features for model input  

### News Data
- Text preprocessing (lowercasing, punctuation removal)  
- Tokenization and stopword removal  
- Sentiment scoring using fine-tuned **BERT**  
- Keyword extraction using **KeyBERT**  

---

## Sentiment Analysis
- Sentiment classification using a **fine-tuned BERT model**  
- Key topics extracted with KeyBERT  
- Word clouds generated to visualize recurring themes  

---

## Visualization and Dashboard
The **Streamlit dashboard** integrates all modules:
- Candlestick charts with real-time OHLC data  
- Moving averages and volume overlays  
- Sentiment trends over time  
- News headlines and keyword frequency  
- Model predictions and evaluation metrics  
- AI chatbot trained on cryptocurrency context  

---

## Forecasting Models

### ARIMA
- Traditional statistical model for linear and stationary time series  
- Good interpretability but limited on non-linear data  

### LSTM
- Captures long-term dependencies in volatile time series  
- Provided the best predictive performance  

### Prophet
- Handles trends, seasonality, and holidays efficiently  
- Robust to missing data and outliers  

### Evaluation Metrics
| Model  | MAE    | MSE       | RMSE   | R² Score |
|--------|--------|-----------|--------|----------|
| ARIMA  | 124.56 | 30678.12  | 175.13 | 0.62     |
| LSTM   | 89.42  | 15899.33  | 126.06 | 0.81     |
| Prophet| 101.85 | 18456.78  | 135.80 | 0.77     |

---

## AI Chatbot Integration
- GPT-based AI assistant integrated into the dashboard  
- Users can query trends, forecasts, or cryptocurrency definitions  
- Fine-tuned with cryptocurrency context for accurate responses  

---

## Conclusion
The system effectively integrates **real-time OHLC data**, **sentiment analysis**, and **predictive modeling** in a single interactive dashboard. LSTM performed best for forecasting due to its ability to model the non-linear, volatile nature of crypto markets.

---

## Future Work
- Incorporate social media sentiment (Twitter, Reddit)  
- Apply Transformer-based time series models (TFT, Informer)  
- Deploy on cloud platforms with GPU acceleration  
- Extend to multi-asset portfolios and portfolio optimization  

---

## References
1. yfinance – Yahoo Finance API for Python  
2. NewsAPI.org – Crypto News Aggregation  
3. Vaswani et al., *Attention is All You Need*, 2017  
4. Facebook Prophet – Open Source Time Series Forecasting  
5. Darts Library – Forecasting Models in Python  
6. KeyBERT – Minimal Keyword Extraction  
7. Streamlit – Interactive Dashboards in Python  
