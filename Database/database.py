import sqlite3
from datetime import date
from pathlib import Path
import streamlit as st
import yfinance as yf
from allsettings import databasefilename


current_file = Path(__file__).resolve()  # finds where this file is
project_folder = current_file.parent.parent  # goes back to the main project folder
database_path = project_folder / "Database" / databasefilename  # connects to the database folder


def create_database():
    conn = sqlite3.connect(database_path)  # connects to the database
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_info (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (ticker, date)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_results (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        run_date TEXT,
        model_name TEXT,
        mae REAL,
        rmse REAL,
        directional_accuracy REAL
    )
    """)
    conn.commit()
    conn.close()  # closes the database connection


def save_model_results(ticker, results_df, direction_accuracy):
    conn = sqlite3.connect(database_path)

    for row in results_df.to_dict("records"): # goes through each model result
        model_name = row["Model"]  # gets the model name
        mae = float(row["MAE"].replace("$", ""))  # removes the dollar sign and makes it a number
        rmse = float(row["RMSE"].replace("$", ""))
        curr_date = date.today()  # gets today's date

        conn.execute("""
        INSERT INTO model_results
        (ticker, run_date, model_name, mae, rmse, directional_accuracy)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, curr_date, model_name, mae, rmse, direction_accuracy))  # saves the model result

    conn.commit()
    conn.close()


def download_stock_data(ticker, start_date):
    df = yf.download(ticker, start=start_date)  # downloads the stock data
    df.reset_index(inplace=True)  # makes the date a normal column
    df.columns = df.columns.get_level_values(0)  # fixes the column names from yfinance
    return df
