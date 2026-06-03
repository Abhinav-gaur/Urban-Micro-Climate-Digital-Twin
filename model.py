# model.py

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_data(df):
    """
    Cleans and prepares dataset for training
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    # Drop missing values
    df = df.dropna()

    if df.empty:
        raise ValueError("Data became empty after dropping missing values.")

    # Convert timestamp if available
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month

    return df


def train_model(X, y):
    """
    Trains RandomForest model
    """

    if len(X) == 0:
        raise ValueError("Feature matrix is empty. Cannot train model.")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)  # 🔥 THIS IS TRAINING

    return model


def evaluate_model(model, X_test, y_test):
    """
    Returns evaluation metrics
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return predictions, metrics


def run_forecast(df):
    """
    1-Hour Ahead Temperature Forecast Pipeline
    """

    df = preprocess_data(df)

    required_columns = ["temperature", "humidity", "wind_speed"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create target: next hour temperature
    df["target_temp"] = df["temperature"].shift(-1)

    df = df.dropna()

    feature_columns = ["temperature", "humidity", "wind_speed"]

    if "hour" in df.columns:
        feature_columns += ["hour", "day", "month"]

    X = df[feature_columns]
    y = df["target_temp"]

    if len(X) < 10:
        raise ValueError("Not enough data to train model.")

    # Chronological split: first 80% train, last 20% test.
    # Random split would cause data leakage for time-series forecasting.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = train_model(X_train, y_train)

    predictions, metrics = evaluate_model(model, X_test, y_test)

    # ---- 1-Hour Forecast ----
    last_row = X.iloc[-1:]
    next_hour_prediction = model.predict(last_row)[0]

    # --- Multi-Step Forecast (Next 6 Hours) ---

    forecast_horizon = 6
    multi_step_predictions = []

    last_known = X.iloc[-1:].copy()

    for step in range(forecast_horizon):
        next_pred = model.predict(last_known)[0]
        multi_step_predictions.append(next_pred)
        
        # Update temperature feature for next prediction
        last_known["temperature"] = next_pred

    forecast_df = pd.DataFrame({
        "Hour": range(1, forecast_horizon + 1),
        "Predicted_Temperature": multi_step_predictions
    })

    return {
    "model": model,
    "test_predictions": predictions,
    "metrics": metrics,
    "next_hour_temp": next_hour_prediction,
    "forecast_df": forecast_df
}