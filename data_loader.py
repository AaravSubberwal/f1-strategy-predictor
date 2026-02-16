import fastf1
import pandas as pd
import os

def load_race_data(year=2023, gp="Monaco", session_type="R"):
    """Download and cache F1 race session data."""

    cache_dir = os.path.join("data", "raw")
    os.makedirs(cache_dir, exist_ok=True)

    fastf1.Cache.enable_cache(cache_dir)

    print(f"Loading {year} {gp} {session_type} session...")
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    laps = session.laps
    print(f"Loaded {len(laps)} laps.")
    return laps

if __name__ == "__main__":
    laps = load_race_data()
    laps.to_csv("data/raw/monaco_2023.csv", index=False)
    print("Data saved to data/raw/monaco_2023.csv")
