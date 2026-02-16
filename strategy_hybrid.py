import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/lap_time_predictor.pkl")

def simulate_strategy(base_features, total_laps=78, pit_laps=[]):
    """
    Simulate a race for a given set of pit laps.
    Returns total predicted race time (seconds).
    """
    laps = []
    current_stint = 1
    compound = base_features["Compound"]

    for lap in range(1, total_laps + 1):
        features = base_features.copy()
        features["LapNumber"] = lap
        features["Stint"] = current_stint
        features["Compound"] = compound

        # Predict lap time
        X = pd.DataFrame([features])
        lap_time = model.predict(X)[0]

        # Add simple tire degradation (lap times increase gradually)
        lap_time += 0.05 * (lap % (total_laps // (len(pit_laps) + 1)))

        laps.append(lap_time)

        # Handle pit stops
        if lap in pit_laps:
            current_stint += 1
            compound = (compound + 1) % 3  # Change tire type

    # Add pit time penalties (25s each)
    total_time = np.sum(laps) + len(pit_laps) * 25
    return total_time


def find_best_single_pit(base_features, total_laps=78):
    """Find the best lap for a single pit-stop strategy."""
    results = []
    for pit_lap in range(10, total_laps - 10, 2):
        total_time = simulate_strategy(base_features, total_laps, [pit_lap])
        results.append((pit_lap, total_time))

    df = pd.DataFrame(results, columns=["PitLap", "TotalTime"])
    best = df.loc[df["TotalTime"].idxmin()]
    return best.PitLap, best.TotalTime, df


def compare_all_strategies(base_features):
    """Compare 0-stop, best 1-stop, and 2-stop strategies."""
    total_laps = 78
    results = {}

    # 0-stop
    no_stop_time = simulate_strategy(base_features, total_laps, [])
    results["0-stop"] = no_stop_time

    # 1-stop (search best lap)
    best_pit, best_time, _ = find_best_single_pit(base_features, total_laps)
    results[f"1-stop (Lap {int(best_pit)})"] = best_time

    # 2-stop (try typical pattern)
    two_stop_time = simulate_strategy(base_features, total_laps, [25, 55])
    results["2-stop (Lap 25, 55)"] = two_stop_time

    # Print comparison
    print("\n🏁 Strategy Comparison (Predicted Total Race Time):")
    for strat, t in results.items():
        print(f"{strat:25s}: {t:.2f} seconds")

    best_strategy = min(results, key=results.get)
    print(f"\n🚀 Optimal Strategy: {best_strategy} 🏆")
    return results, best_strategy


if __name__ == "__main__":
    # Define baseline features (average driver/team/conditions)
    base_features = {
        "LapNumber": 1,
        "Stint": 1,
        "Compound": 1,     # 0: Hard, 1: Medium, 2: Soft (encoded)
        "Driver": 5,
        "Team": 3,
        "TrackStatus": 1
    }

    results, best = compare_all_strategies(base_features)

    # Save results for dashboard
    df = pd.DataFrame(list(results.items()), columns=["Strategy", "TotalTime"])
    df.to_csv("data/strategy_comparison.csv", index=False)
    print("✅ Saved results to data/strategy_comparison.csv")
