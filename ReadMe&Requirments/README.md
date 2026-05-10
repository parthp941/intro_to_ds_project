## Stock Prediction Model

This project is a Streamlit app that predicts the next trading day stock price for a ticker symbol entered by the user. It also compares basic OHLCV features against technical indicator features.

## How it works
. Downloads stock data from yfinance
. Creates feature engineering like moving averages, RSI, MACD, and Bollinger Bands
. Uses K-Means clustering to create market regime labels
. Trains machine learning models
. Compares Linear Regression, Random Forest, Gradient Boosting, and an Average Model
. Uses TimeSeriesSplit walk-forward cross validation
. Shows feature importance from the Random Forest model
. Predicts whether the stock may go up or down
. Saves model results into a SQLite database
. Displays charts showing prediction errors and model results

## How To Run
pip install -r ReadMe\&Requirments/requirements.txt
streamlit run App/app.py

## How To Create Research Result Tables
python Experiments/run_research_results.py
