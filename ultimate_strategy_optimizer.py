import pandas as pd
import numpy as np
import joblib
import itertools

# Load the trained lap time prediction model
model = joblib.load("models/lap_time_predictor.pkl")

def simulate_strategy(base_features, total_laps=78, pit_laps=None):
    """
    Simulate race total time for a given list of pit laps.
    Returns total predicted race time (seconds).
    """
    if pit_laps is None:
        pit_laps = []

    laps = []
    current_stint = 1
    compound = base_features["Compound"]

    for lap in range(1, total_laps + 1):
        features = base_features.copy()
        features["LapNumber"] = lap
        features["Stint"] = current_stint
        features["Compound"] = compound

        # Predict base lap time
        X = pd.DataFrame([features])
        lap_time = model.predict(X)[0]

        # Simulate tire degradation (lap times increase gradually)
        lap_time += 0.05 * (lap % (total_laps // (len(pit_laps) + 1)))

        laps.append(lap_time)

        # If a pit stop occurs, switch stint/compound
        if lap in pit_laps:
            current_stint += 1
            compound = (compound + 1) % 3  # change tire compound

    # Add pit stop penalty (25 seconds per stop)
    total_time = np.sum(laps) + len(pit_laps) * 25
    return total_time


def find_best_strategy(base_features, total_laps=78, max_stops=3):
    """
    Try different numbers of pit stops (0 to max_stops) and find
    the best combination of pit laps for minimum race time.
    """
    best_time = float("inf")
    best_strategy = None
    results = []

    # Try all possible pit stop counts
    for stops in range(0, max_stops + 1):
        # No pit stop case
        if stops == 0:
            total_time = simulate_strategy(base_features, total_laps, [])
            results.append((stops, [], total_time))
            if total_time < best_time:
                best_time, best_strategy = total_time, (stops, [])
            continue

        # Generate all combinations of pit laps (rough grid search)
        possible_laps = range(10, total_laps - 10, 10)
        for pit_laps in itertools.combinations(possible_laps, stops):
            total_time = simulate_strategy(base_features, total_laps, list(pit_laps))
            results.append((stops, pit_laps, total_time))

            if total_time < best_time:
                best_time = total_time
                best_strategy = (stops, pit_laps)

    return best_strategy, best_time, results


if __name__ == "__main__":
    base_features = {
        "LapNumber": 1,
        "Stint": 1,
        "Compound": 1,     # 0: Hard, 1: Medium, 2: Soft
        "Driver": 5,
        "Team": 3,
        "TrackStatus": 1
    }

    print("🏎️ Evaluating pit strategies... this may take a few seconds.\n")
    best_strategy, best_time, results = find_best_strategy(base_features, total_laps=78, max_stops=3)

    # Display all tested strategies
    df = pd.DataFrame(results, columns=["Stops", "PitLaps", "TotalTime"])
    df_sorted = df.sort_values(by="TotalTime").reset_index(drop=True)

    print("\n🏁 Top 5 Best Strategies:")
    print(df_sorted.head(5))

    # Final best strategy
    stops, pit_laps = best_strategy
    print("\n🚀 Optimal Strategy Found:")
    print(f"👉 Number of pit stops: {stops}")
    print(f"👉 Pit on laps: {pit_laps}")
    print(f"👉 Predicted Total Race Time: {best_time:.2f} seconds")

    # Save results
    df_sorted.to_csv("data/final_strategy_results.csv", index=False)
    print("\n✅ All strategy results saved to data/final_strategy_results.csv")
