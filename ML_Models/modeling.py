import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def find_gradient_boosting_features(X_train):
    gradient_boosting_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MovingAverage_50",
        "MovingAverage_200",
        "Daily_Return",
        "Volatility",
    ]
    available_gradient_boosting_features = []
    for feature in gradient_boosting_features:
        if feature in X_train.columns:
            available_gradient_boosting_features.append(feature)
    return available_gradient_boosting_features


def price_metrics(actual_prices, predicted_prices):
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    return mae, rmse


def run_walk_forward_validation(df, features):
    X = df[features]
    y_return = df["Target_Return"]
    y_price = df["Target"]
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_return_train = y_return.iloc[train_index]
        y_price_test = y_price.iloc[test_index]

        test_close_prices = np.asarray(X_test["Close"]).reshape(-1)
        actual_prices = np.asarray(y_price_test).reshape(-1)
        gradient_boosting_features = find_gradient_boosting_features(X_train)

        lr_model = make_pipeline(StandardScaler(), LinearRegression())
        lr_model.fit(X_train, y_return_train)
        lr_return_predictions = lr_model.predict(X_test)
        lr_price_predictions = test_close_prices * (1 + lr_return_predictions)

        rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)
        rf_model.fit(X_train, y_return_train)
        rf_return_predictions = rf_model.predict(X_test).flatten()
        rf_price_predictions = test_close_prices * (1 + rf_return_predictions)

        gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42)
        gb_model.fit(X_train[gradient_boosting_features], y_return_train)
        gb_return_predictions = gb_model.predict(X_test[gradient_boosting_features]).flatten()
        gb_price_predictions = test_close_prices * (1 + gb_return_predictions)

        average_predictions = (lr_price_predictions + rf_price_predictions + gb_price_predictions) / 3

        fold_predictions = {
            "Linear Regression": lr_price_predictions,
            "Random Forest": rf_price_predictions,
            "Gradient Boosting": gb_price_predictions,
            "Average Model": average_predictions,
        }

        for model_name, predicted_prices in fold_predictions.items():
            mae, rmse = price_metrics(actual_prices, predicted_prices)
            rows.append([model_name, mae, rmse])

    walk_forward_df = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE"])
    walk_forward_summary = (
        walk_forward_df
        .groupby("Model")
        .agg({"MAE": ["mean", "std"], "RMSE": ["mean", "std"]})
        .reset_index()
    )
    walk_forward_summary.columns = ["Model", "Mean MAE", "MAE Std", "Mean RMSE", "RMSE Std"]

    for column in ["Mean MAE", "MAE Std", "Mean RMSE", "RMSE Std"]:
        walk_forward_summary[column] = walk_forward_summary[column].round(2)

    return walk_forward_summary


def train_all_models(df, features, use_walk_forward=False):
    X = df[features]
    y_price = df["Target"]
    y_return = df["Target_Return"]
    y_direction = df["Target_Direction"]

    #Test/train split
    X_train, X_test, y_price_train, y_price_test, y_return_train, y_return_test, y_direction_train, y_direction_test = train_test_split(X,y_price,y_return,y_direction,test_size=0.2,shuffle=False)

    # turns all test values into a 1d array so the models can compare properly
    test_close_prices = np.asarray(X_test['Close']).reshape(-1)
    actual_prices = np.asarray(y_price_test).reshape(-1)
    actual_direction = np.asarray(y_direction_test).reshape(-1)

    # baseline where tomorrows price is same as todays closing price
    naive_predictions = test_close_prices
    naive_mae = mean_absolute_error(actual_prices, naive_predictions) #find mae for naive
    naive_rmse = np.sqrt(mean_squared_error(actual_prices, naive_predictions)) #find rmse for naive

    # create Linear Regression model
    lr_model = make_pipeline(StandardScaler(), LinearRegression())
    lr_model.fit(X_train, y_return_train)

    #predictions on testing set
    lr_return_predictions = lr_model.predict(X_test)
    lr_predictions = test_close_prices * (1 + lr_return_predictions)
    lr_predictions = np.asarray(lr_predictions).reshape(-1)

    # predict tomorrows price
    latest_data = df[features].iloc[[-1]]
    latest_close = df['Close'].iloc[-1].item()

    lr_latest_return = lr_model.predict(latest_data)
    lr_latest_return = np.asarray(lr_latest_return).reshape(-1)[0]
    lr_tomorrow_prediction = latest_close * (1 + lr_latest_return)

    lr_mae = mean_absolute_error(actual_prices, lr_predictions)
    lr_rmse = np.sqrt(mean_squared_error(actual_prices, lr_predictions))

    #1). Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)

    rf_model.fit(X_train, y_return_train)

    # predict test returns
    rf_return_preds = rf_model.predict(X_test).flatten()

    rf_price_preds = test_close_prices * (1 + rf_return_preds)

    # predict tomorrow's return and price
    rf_latest_return = rf_model.predict(latest_data).flatten()[0]
    rf_tomorrow_prediction = latest_close * (1 + rf_latest_return)

    # see model performance
    rf_mae = mean_absolute_error(actual_prices, rf_price_preds)


    rf_rmse = np.sqrt(mean_squared_error(actual_prices, rf_price_preds))


    #2). Gradient Boosting
    available_gradient_boosting_features = find_gradient_boosting_features(X_train)
    gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, random_state=42)

    gb_model.fit(X_train[available_gradient_boosting_features], y_return_train)

    #predict test returns
    gb_return_preds = gb_model.predict(X_test[available_gradient_boosting_features]).flatten()
    gb_price_preds = test_close_prices * (1 + gb_return_preds)

    #predict tomorrow's return and price
    gb_latest_return = gb_model.predict(latest_data[available_gradient_boosting_features]).flatten()[0]

    gb_tomorrow_prediction = latest_close * (1 + gb_latest_return)

    #see model performance
    gb_mae = mean_absolute_error(actual_prices, gb_price_preds)
    gb_rmse = np.sqrt(mean_squared_error(actual_prices, gb_price_preds))


    #Average for all models
    average_predictions = (lr_predictions + rf_price_preds + gb_price_preds) / 3
    average_tomorrow_prediction = (lr_tomorrow_prediction + rf_tomorrow_prediction + gb_tomorrow_prediction) / 3
    average_mae = mean_absolute_error(actual_prices, average_predictions) #MAE between actual vs predicted
    average_rmse = np.sqrt(mean_squared_error(actual_prices, average_predictions)) #RMSE for actual vs predicted


    #3). Direction Prediction (UP/DOWN)
    direction_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    direction_model.fit(X_train, y_direction_train)

    # predict tomorrow direction
    direction = direction_model.predict(latest_data)[0]
    confidence = direction_model.predict_proba(latest_data)
    if direction == True:
        up_down = "Up"
        confidence = confidence[0][1] * 100
    else:
        up_down = "Down"
        confidence = confidence[0][0] * 100

    # tests direction predictions on the testing set
    direction_test_predictions = direction_model.predict(X_test)
    direction_accuracy = accuracy_score(actual_direction, direction_test_predictions)

    #creates dataframe to display in streamlit
    results_df = pd.DataFrame(
        [
            ["Baseline", f"${latest_close:.2f}", f"${naive_mae:.2f}", f"${naive_rmse:.2f}"],
            ["Linear Regression", f"${lr_tomorrow_prediction:.2f}", f"${lr_mae:.2f}", f"${lr_rmse:.2f}"],
            ["Random Forest", f"${rf_tomorrow_prediction:.2f}", f"${rf_mae:.2f}", f"${rf_rmse:.2f}"],
            ["Gradient Boosting", f"${gb_tomorrow_prediction:.2f}", f"${gb_mae:.2f}", f"${gb_rmse:.2f}"],
            ["Average Model", f"${average_tomorrow_prediction:.2f}", f"${average_mae:.2f}", f"${average_rmse:.2f}"],
        ],
        columns=["Model", "Next Trading Day Prediction", "MAE", "RMSE"],
    )

    #shows best model based on ticker
    model_names = ["Linear Regression", "Random Forest", "Gradient Boosting", "Average Model"]
    mae_values = [lr_mae, rf_mae, gb_mae, average_mae]

    best_model_index = mae_values.index(min(mae_values)) #assigns the lowest mae to best model index
    best_model = model_names[best_model_index] #takes the index from models names to find the best model
    best_mae = mae_values[best_model_index] #takes the mae from the best model

    predictions = {
        "Linear Regression": lr_predictions,
        "Random Forest": rf_price_preds,
        "Gradient Boosting": gb_price_preds,
        "Average Model": average_predictions,
    }

    feature_importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    outputs = {
        "X_test": X_test,
        "actual_prices": actual_prices,
        "latest_close": latest_close,
        "results_df": results_df,
        "best_model": best_model,
        "best_mae": best_mae,
        "direction_accuracy": direction_accuracy,
        "up_down": up_down,
        "confidence": confidence,
        "rf_model": rf_model,
        "feature_importance_df": feature_importance_df,
    }

    if use_walk_forward:
        outputs["walk_forward_df"] = run_walk_forward_validation(df, features)

    return outputs, predictions
