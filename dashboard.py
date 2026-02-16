import streamlit as st
import pandas as pd
import joblib
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ---------------- Load Model and Mappings ---------------- #

@st.cache_resource
def load_model():
    model_path = "models/lap_time_predictor.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please run `src/model_train.py` first to generate it.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_mappings():
    mapping_path = "data/category_mappings.json"
    if not os.path.exists(mapping_path):
        st.error("❌ Mapping file not found! Please run `src/preprocess.py` first to generate it.")
        st.stop()
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
    driver_name_to_id = {v: int(k) for k, v in mappings["Driver"].items()}
    team_name_to_id = {v: int(k) for k, v in mappings["Team"].items()}
    return driver_name_to_id, team_name_to_id

model = load_model()
driver_name_to_id, team_name_to_id = load_mappings()

# ---------------- Core Simulation Logic ---------------- #

def simulate_strategy(base_features, total_laps=78, pit_laps=None):
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

        X = pd.DataFrame([features])
        lap_time = model.predict(X)[0]
        lap_time += 0.05 * (lap % (total_laps // (len(pit_laps) + 1)))  # degradation
        laps.append(lap_time)

        if lap in pit_laps:
            current_stint += 1
            compound = (compound + 1) % 3  # switch tire compound

    total_time = np.sum(laps) + len(pit_laps) * 25  # 25s pit penalty
    return total_time

def find_best_strategy(base_features, total_laps=78, max_stops=3):
    best_time = float("inf")
    best_strategy = None
    results = []

    for stops in range(0, max_stops + 1):
        if stops == 0:
            total_time = simulate_strategy(base_features, total_laps, [])
            results.append((stops, [], total_time))
            if total_time < best_time:
                best_time, best_strategy = total_time, (stops, [])
            continue

        possible_laps = range(10, total_laps - 10, 10)
        for pit_laps in itertools.combinations(possible_laps, stops):
            total_time = simulate_strategy(base_features, total_laps, list(pit_laps))
            results.append((stops, pit_laps, total_time))
            if total_time < best_time:
                best_time = total_time
                best_strategy = (stops, pit_laps)

    return best_strategy, best_time, results

def generate_lap_time_curve(base_features, best_strategy, total_laps=78):
    """Generate predicted lap times for the best strategy."""
    pit_laps = list(best_strategy[1])
    laps = []
    compounds = []
    current_stint = 1
    compound = base_features["Compound"]

    for lap in range(1, total_laps + 1):
        features = base_features.copy()
        features["LapNumber"] = lap
        features["Stint"] = current_stint
        features["Compound"] = compound

        X = pd.DataFrame([features])
        lap_time = model.predict(X)[0]
        lap_time += 0.05 * (lap % (total_laps // (len(pit_laps) + 1)))

        laps.append(lap_time)
        compounds.append(compound)

        if lap in pit_laps:
            current_stint += 1
            compound = (compound + 1) % 3

    df_curve = pd.DataFrame({
        "Lap": range(1, total_laps + 1),
        "LapTime": laps,
        "Compound": compounds
    })
    return df_curve, pit_laps

# ---------------- Streamlit Dashboard ---------------- #

st.set_page_config(page_title="🏎️ F1 Strategy Optimizer", layout="wide")
st.title("🏁 F1 Pit-Stop Strategy Optimizer")
st.markdown("### Predict optimal pit stops using machine learning and race simulation 🚀")

# Sidebar
st.sidebar.header("Race Settings")
total_laps = st.sidebar.slider("Total Laps", 50, 100, 78)
max_stops = st.sidebar.slider("Max Pit Stops to Test", 0, 3, 3)
st.sidebar.markdown("---")

driver_name = st.sidebar.selectbox("Select Driver", sorted(driver_name_to_id.keys()))
team_name = st.sidebar.selectbox("Select Team", sorted(team_name_to_id.keys()))
compound = st.sidebar.selectbox("Starting Tire Compound", ["Hard (0)", "Medium (1)", "Soft (2)"], index=1)
track_status = st.sidebar.number_input("Track Status", min_value=1, max_value=5, value=1)

if st.button("🔍 Find Best Strategy"):
    with st.spinner("Simulating race strategies..."):
        base_features = {
            "LapNumber": 1,
            "Stint": 1,
            "Compound": int(compound[-2]),
            "Driver": driver_name_to_id[driver_name],
            "Team": team_name_to_id[team_name],
            "TrackStatus": track_status
        }

        best_strategy, best_time, results = find_best_strategy(base_features, total_laps, max_stops)
        df = pd.DataFrame(results, columns=["Stops", "PitLaps", "TotalTime"])
        df["PitLaps"] = df["PitLaps"].apply(lambda x: ', '.join(map(str, x)) if x else "None")
        df = df.sort_values(by="TotalTime").reset_index(drop=True)

        st.success(f"🏆 **Optimal Strategy:** {best_strategy[0]} stops | Laps: {best_strategy[1]} | "
                   f"Total Time: {best_time:.2f} sec")

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.markdown("#### 🏎️ Top 10 Fastest Strategies")
            df["PitLaps"] = df["PitLaps"].astype(str)
            fig = px.bar(df.head(10),
                x="PitLaps",
                y="TotalTime",
                color="Stops",
                text_auto=".2f",
                title="Top 10 Fastest Strategies")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Top Strategies (Table)")
            st.dataframe(df.head(10), use_container_width=True)

        st.download_button(
            label="📥 Download Results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="strategy_results.csv",
            mime="text/csv"
        )

        # ---------- Lap Time Curve Visualization ----------
        df_curve, pit_laps = generate_lap_time_curve(base_features, best_strategy, total_laps)
        st.markdown("#### ⏱️ Predicted Lap Time Curve for Best Strategy")

        compound_colors = {0: "white", 1: "yellow", 2: "red"}  # Hard, Medium, Soft
        df_curve["CompoundColor"] = df_curve["Compound"].map(compound_colors)

        fig2 = go.Figure()

        # Plot each compound stint separately
        for comp, group in df_curve.groupby("Compound"):
            color = compound_colors[comp]
            fig2.add_trace(go.Scatter(
                x=group["Lap"],
                y=group["LapTime"],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"{['Hard', 'Medium', 'Soft'][comp]}"
            ))

        # Mark pit stops
        for lap in pit_laps:
            fig2.add_vline(x=lap, line_dash="dash", line_color="gray")

        fig2.update_layout(
            title="Lap Time Progression Across Race",
            xaxis_title="Lap Number",
            yaxis_title="Predicted Lap Time (s)",
            template="plotly_dark"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Tire color legend
        st.markdown("""
        **🟡 Tire Compounds Legend:**
        - ⚪ Hard (White)
        - 🟡 Medium (Yellow)
        - 🔴 Soft (Red)
        """)

else:
    st.info("👈 Adjust parameters on the left and click 'Find Best Strategy' to start simulation.")
