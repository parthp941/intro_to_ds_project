from pathlib import Path
import sys

import pandas as pd

current_file = Path(__file__).resolve()
project_folder = current_file.parent.parent
sys.path.append(str(project_folder))

from allsettings import allfeatures, basicfeatures, research_tickers, startdate
from Database.database import download_stock_data
from FeatureEngineering.feature_engineering import allfeatureengineering
from ML_Models.modeling import run_walk_forward_validation, train_all_models


def money_to_float(value):
    return float(str(value).replace("$", ""))


def add_model_rows(ticker, feature_set_name, model_output):
    rows = []
    for row in model_output["results_df"].to_dict("records"):
        rows.append(
            {
                "ticker": ticker,
                "feature_set": feature_set_name,
                "model": row["Model"],
                "next_trading_day_prediction": money_to_float(row["Next Trading Day Prediction"]),
                "mae": money_to_float(row["MAE"]),
                "rmse": money_to_float(row["RMSE"]),
                "directional_accuracy": model_output["direction_accuracy"],
                "best_model": model_output["best_model"],
                "best_mae": model_output["best_mae"],
            }
        )
    return rows


def main():
    results_folder = project_folder / "Results"
    results_folder.mkdir(exist_ok=True)

    all_rows = []
    walk_forward_rows = []
    feature_sets = {
        "Basic Features": basicfeatures,
        "Technical Indicator Features": allfeatures,
    }

    for ticker in research_tickers:
        print(f"Running {ticker}")
        raw_df = download_stock_data(ticker, startdate)
        df = allfeatureengineering(raw_df)

        for feature_set_name, features in feature_sets.items():
            model_output, _ = train_all_models(df, features)
            all_rows.extend(add_model_rows(ticker, feature_set_name, model_output))

            walk_forward_df = run_walk_forward_validation(df, features)
            walk_forward_df.insert(0, "feature_set", feature_set_name)
            walk_forward_df.insert(0, "ticker", ticker)
            walk_forward_rows.extend(walk_forward_df.to_dict("records"))

    results_df = pd.DataFrame(all_rows)
    walk_forward_results_df = pd.DataFrame(walk_forward_rows)

    results_df.to_csv(results_folder / "multi_ticker_results.csv", index=False)
    walk_forward_results_df.to_csv(results_folder / "walk_forward_results.csv", index=False)

    print(f"Saved results to {results_folder}")


main()
