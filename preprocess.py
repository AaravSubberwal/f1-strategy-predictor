import pandas as pd
import numpy as np
import os
import json

def preprocess_laps(laps_df):
    df = laps_df.copy()

    # Convert LapTime to seconds
    if not np.issubdtype(df["LapTime"].dtype, np.timedelta64):
        df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
    df = df.dropna(subset=["LapTime"])
    df["LapTime(s)"] = df["LapTime"].dt.total_seconds()

    # --- Encode categorical columns ---
    mappings = {}
    for col in ["Driver", "Team", "Compound"]:
        df[col] = df[col].astype("category")
        mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    features = ["LapNumber", "Stint", "Compound", "Driver", "Team", "TrackStatus", "LapTime(s)"]
    df = df[features].dropna()

    print(f"✅ Preprocessed {len(df)} laps.")
    return df, mappings


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv("data/raw/monaco_2023.csv")
    df_clean, mappings = preprocess_laps(df)

    df_clean.to_csv("data/processed_monaco.csv", index=False)
    print("✅ Saved processed data to data/processed_monaco.csv")

    with open("data/category_mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)
    print("✅ Saved driver/team name mappings to data/category_mappings.json")
