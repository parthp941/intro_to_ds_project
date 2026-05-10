from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# finds the main project folder so app.py can use files from other folders
current_file = Path(__file__).resolve() #finds the current path for this file
project_folder = current_file.parent.parent #finds main folder from this file

sys.path.append(str(project_folder))

# imports everything from allsettings important settings like the database file, features list, and start date
from allsettings import allfeatures, basicfeatures, startdate
# imports all sql functions from database
from Database.database import (create_database, download_stock_data, save_model_results)
# imports feature enginering function
from FeatureEngineering.feature_engineering import allfeatureengineering
# imports model training functions
from ML_Models.modeling import train_all_models
# Imports graph functions
from App.visualizations import plot_histogram, plot_line_chart


def money_to_float(value):
    return float(str(value).replace("$", ""))


def get_model_row(model_output, model_name):
    results_df = model_output["results_df"]
    return results_df[results_df["Model"] == model_name].iloc[0]


def create_feature_comparison_df(basic_output, technical_output):
    basic_best_row = get_model_row(basic_output, basic_output["best_model"])
    technical_best_row = get_model_row(technical_output, technical_output["best_model"])

    return pd.DataFrame(
        [
            [
                "Basic Features",
                basic_output["best_model"],
                basic_best_row["Next Trading Day Prediction"],
                basic_best_row["MAE"],
                basic_best_row["RMSE"],
            ],
            [
                "Technical Indicator Features",
                technical_output["best_model"],
                technical_best_row["Next Trading Day Prediction"],
                technical_best_row["MAE"],
                technical_best_row["RMSE"],
            ],
        ],
        columns=["Feature Set", "Best Model", "Best Model Prediction", "Best MAE", "Best RMSE"],
    )


def main():
    st.title("Stock Price Prediction App")
    create_database() #call function
    ticker = st.text_input("Enter a stock ticker symbol").upper() #users imput what stock they want

    if st.button("Predict"):
        if not ticker:
            st.error("Please enter a stock ticker symbol.")
            return
        raw_df = download_stock_data(ticker, startdate) #gathers data from yfinance api based on ticker
        if raw_df.empty:
            st.error("Invalid Ticker Symbol")
            return

        #takes rawdf and applies everything from feature engineering function to it
        df = allfeatureengineering(raw_df)

        # trains the models and stores it in modeloutput
        basic_model_output, _ = train_all_models(df, basicfeatures)
        model_output, predictions = train_all_models(df, allfeatures, use_walk_forward=True)

        # saves results in database
        save_model_results(ticker, model_output["results_df"], model_output["direction_accuracy"])

        # gets latest date in dataset
        latest_date = df["Date"].iloc[-1].date()

        st.subheader(f"{ticker} Predicted Next Trading Day Price")

        col1, col2, col3 = st.columns(3) #set columns for streamlit

        # average price from all models
        results_df = model_output["results_df"]
        average_model_row = results_df[results_df["Model"] == "Average Model"]
        average_model_price = average_model_row["Next Trading Day Prediction"].iloc[0]

        col1.metric("Average Model Prediction", average_model_price)
        col2.metric("Latest Close", f"${model_output['latest_close']:.2f}")
        col3.metric("Latest Data Date", str(latest_date))


        st.subheader("Predicted Price Direction")
        st.write(f"The direction model predicts that {ticker} will go {model_output['up_down']}.")
        st.write(f"Confidence: {model_output['confidence']:.2f}%")

        st.subheader("Model Results")
        st.dataframe(model_output["results_df"])
        st.subheader(
            f"Best machine learning model for {ticker}: {model_output['best_model']} " #show best model
            f"with an MAE of ${model_output['best_mae']:.2f}"
        )

        st.subheader("Basic vs Technical Indicator Features")
        feature_comparison_df = create_feature_comparison_df(basic_model_output, model_output)
        st.dataframe(feature_comparison_df)

        chart_df = pd.DataFrame(
            {
                "Feature Set": feature_comparison_df["Feature Set"],
                "Best MAE": feature_comparison_df["Best MAE"].apply(money_to_float),
            }
        ).set_index("Feature Set")
        st.bar_chart(chart_df)

        st.subheader("Walk-Forward Cross Validation")
        st.dataframe(model_output["walk_forward_df"])

        st.subheader("K-Means Market Regimes")
        regime_counts = df["Market_Regime"].value_counts().sort_index()
        st.dataframe(regime_counts.rename("Days in Regime"))
        st.bar_chart(regime_counts)


        st.subheader("Price, RSI, and Bollinger Bands Graphs")

        recent_df = df.tail(250) #use last 250 days

        plot_line_chart(
            f"{ticker} Close Price with Moving Averages",
            recent_df["Date"],
            {
                "Close": recent_df["Close"],
                "50-Day Moving Average": recent_df["MovingAverage_50"],
                "200-Day Moving Average": recent_df["MovingAverage_200"],
            },
            "Price in $",
        )

        # RSI graph
        plot_line_chart(
            f"{ticker} RSI",
            recent_df["Date"],
            {"RSI": recent_df["RSI"]},
            "RSI",
        )

        # bollinger Bands graph
        plot_line_chart(
            f"{ticker} Bollinger Bands",
            recent_df["Date"],
            { "Close": recent_df["Close"],
                "Upper Band": recent_df["BB_Upper"],
                "Lower Band": recent_df["BB_Lower"],
            },
            "Price in $",
        )


        st.subheader("Actual vs Predicted Prices")

        # gets dates for last 100 predictions
        test_rows = model_output["X_test"].index
        test_dates = df.loc[test_rows, "Date"]
        plot_dates = test_dates.tail(100)

        # Plots actual prices compared to predicted prices
        plot_line_chart(
            f"{ticker} Actual vs Predicted Prices",
            plot_dates,
            { "Actual": model_output["actual_prices"][-100:], #finds for last 100 days
                "Linear Regression": predictions["Linear Regression"][-100:],
                "Random Forest": predictions["Random Forest"][-100:],
                "Gradient Boosting": predictions["Gradient Boosting"][-100:],
                "Average Model": predictions["Average Model"][-100:],
            },
            "Price in $",
        )


        st.subheader("Prediction Error Graphs")

        actual_prices = model_output["actual_prices"]

        plot_line_chart(
            f"{ticker} Prediction Errors",
            plot_dates,
            { #subtracts actual and predictions for last 100 days
                "Linear Regression Error": actual_prices[-100:] - predictions["Linear Regression"][-100:],
                "Random Forest Error": actual_prices[-100:] - predictions["Random Forest"][-100:],
                "Gradient Boosting Error": actual_prices[-100:] - predictions["Gradient Boosting"][-100:],
                "Average Model Error": actual_prices[-100:] - predictions["Average Model"][-100:],
            },
            "Prediction Error in $",
        )

        plot_histogram(
            f"{ticker} Average Model Error Distribution",
            actual_prices - predictions["Average Model"],
            "Prediction Error in $",
        )

        st.subheader("Feature Importance")
        feature_importance_df = model_output["feature_importance_df"].head(10)
        st.dataframe(feature_importance_df)
        st.bar_chart(feature_importance_df.set_index("Feature")["Importance"])


main() #call function
