import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_lap_time_model(data_path="data/processed_monaco.csv"):
    """Train a regression model to predict lap times."""
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=["LapTime(s)"])
    y = df["LapTime(s)"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"✅ Model trained successfully!")
    print(f"📊 MAE: {mae:.3f} seconds | R²: {r2:.3f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/lap_time_predictor.pkl")
    print("💾 Model saved to models/lap_time_predictor.pkl")

    return model

if __name__ == "__main__":
    train_lap_time_model()
