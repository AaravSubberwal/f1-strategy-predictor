import pandas as pd
import numpy as np
import joblib

def simulate_strategy(model_path="models/lap_time_predictor.pkl",
                      race_laps=78,
                      pit_laps=[35],
                      base_features=None):
    """
    Simulate total race time for a given pit-stop strategy.
    - pit_laps: list of lap numbers where pit stops occur
    - base_features: dict with fixed features like driver, team, etc.
    """

    model = joblib.load(model_path)

    if base_features is None:
        base_features = {
            "Stint": 1,
            "Compound": 0,  # e.g., hard
            "Driver": 0,
            "Team": 0,
            "TrackStatus": 1
        }

    # Prepare lap-wise data
    laps = []
    current_stint = 1

    for lap in range(1, race_laps + 1):
        # Update stint & compound after each pit stop
        if lap in pit_laps:
            current_stint += 1
            # Simulate tire change (e.g., switch compound)
            base_features["Compound"] = (base_features["Compound"] + 1) % 3

        # Degradation effect (lap time slightly increases per lap)
        degradation_factor = 0.05 * (lap % (race_laps // (len(pit_laps)+1)))

        row = {
            "LapNumber": lap,
            "Stint": current_stint,
            "Compound": base_features["Compound"],
            "Driver": base_features["Driver"],
            "Team": base_features["Team"],
            "TrackStatus": base_features["TrackStatus"]
        }

        laps.append(row)

    df = pd.DataFrame(laps)
    preds = model.predict(df)
    total_time = preds.sum() + (len(pit_laps) * 25)  # +25 sec per pit stop

    return total_time, preds


def compare_strategies():
    """Compare a few possible pit strategies."""
    strategies = {
        "1-stop (Lap 40)": [40],
        "2-stop (Lap 25, 55)": [25, 55],
        "No stop": []
    }

    results = {}
    for name, stops in strategies.items():
        total_time, _ = simulate_strategy(pit_laps=stops)
        results[name] = total_time

    print("\n🏁 Strategy Comparison (Predicted Total Race Time):")
    for strat, t in results.items():
        print(f"{strat:20s}: {t:.2f} seconds")

    best = min(results, key=results.get)
    print(f"\n🚀 Optimal Strategy: {best} 🏆")
    return results


if __name__ == "__main__":
    compare_strategies()
