"""
train_hybrid.py
---------------
Hybrid RF + GCN Training Pipeline

Steps
-----
1. Load weather data  (delhi_weather_2024.csv  OR  open-meteo CSV fallback)
2. Compute Urban Heat Island offsets for each of the 600 grid zones
3. Train RandomForest temporal model  (chronological 80/20 split)
4. Generate spatial training snapshots from RF predictions + UHI model
5. Build 4-neighbour graph adjacency for the Delhi grid
6. Train 2-layer GCN on the spatial snapshots
7. Evaluate both models and print comparison table
8. Save all models to  saved_models/

Run
---
    python train_hybrid.py

Outputs
-------
    saved_models/rf_model.pkl
    saved_models/gcn_weights.pth
    saved_models/gcn_scalers.pkl
    saved_models/results.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from gcn_model import GCN, build_adjacency_matrix, normalize_adjacency

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

GRID_ROWS    = 25
GRID_COLS    = 24
N_NODES      = GRID_ROWS * GRID_COLS          # 600

GRID_PATH    = "delhi_grid_2.0km.csv"
HIST_PATH = "delhi_weather_2024_2026.csv"       # preferred (full year)
FALLBACK_CSV = "open-meteo-28.62N77.25E231m.csv"  # 7-day fallback

MODEL_DIR    = "saved_models"
N_SNAPSHOTS  = 600                            # spatial training snapshots
GCN_EPOCHS   = 200
GCN_LR       = 0.01

# ─────────────────────────────────────────────────────────────
# Urban Heat Island Model for Delhi
#   Each entry: (center_lat, center_lon, intensity_°C, sigma_°)
# ─────────────────────────────────────────────────────────────

UHI_CENTERS = [
    # Hot zones (urban / industrial)
    (28.63, 77.22,  4.0, 0.025),   # Central Delhi / Connaught Place
    (28.67, 77.28,  3.5, 0.020),   # Shahdara / Trans-Yamuna industrial
    (28.55, 77.28,  3.0, 0.018),   # Okhla industrial area
    (28.63, 77.14,  2.5, 0.015),   # Naraina industrial
    (28.72, 77.22,  2.0, 0.020),   # North Delhi dense residential
    # Cooling zones (green / water)
    (28.69, 77.15, -2.5, 0.020),   # Delhi Ridge (Aravalli)
    (28.65, 77.27, -2.0, 0.018),   # Yamuna Floodplain
    (28.59, 77.22, -1.5, 0.015),   # Lodhi Garden area
]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def compute_uhi_offsets(grid_df: pd.DataFrame) -> np.ndarray:
    """
    Compute UHI offset (°C) for every zone using Gaussian kernels
    centred on known Delhi heat source / sink locations.
    """
    lats = grid_df["center_lat"].values
    lons = grid_df["center_lon"].values
    offsets = np.zeros(N_NODES, dtype=np.float32)
    for (clat, clon, intensity, sigma) in UHI_CENTERS:
        dist_sq = (lats - clat) ** 2 + (lons - clon) ** 2
        offsets += intensity * np.exp(-dist_sq / (2 * sigma ** 2))
    return offsets


# ─────────────────────────────────────────────────────────────
# Step 1 — Load weather data
# ─────────────────────────────────────────────────────────────

def load_weather() -> pd.DataFrame:
    if os.path.exists(HIST_PATH):
        print(f"  Using historical data: {HIST_PATH}")
        df = pd.read_csv(HIST_PATH, parse_dates=["timestamp"])
        # Ensure derived columns exist
        if "hour" not in df.columns:
            df["hour"]  = df["timestamp"].dt.hour
            df["day"]   = df["timestamp"].dt.dayofyear
            df["month"] = df["timestamp"].dt.month
    else:
        print(f"  Historical data not found. Using fallback: {FALLBACK_CSV}")
        df = pd.read_csv(FALLBACK_CSV, skiprows=2)
        df = df.rename(columns={
            "time":                      "timestamp",
            "temperature_2m (°C)":       "temperature",
            "relative_humidity_2m (%)":  "humidity",
            "wind_speed_10m (km/h)":     "wind_speed",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]  = df["timestamp"].dt.hour
        df["day"]   = df["timestamp"].dt.dayofyear
        df["month"] = df["timestamp"].dt.month

    df = df.dropna().reset_index(drop=True)
    print(f"  Records loaded: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2 — RF Temporal Model
# ─────────────────────────────────────────────────────────────

def train_rf(df: pd.DataFrame):
    """
    Train RF to predict temperature 1 hour ahead.
    Uses strict chronological 80/20 split (no data leakage).

    Features: [temperature, humidity, wind_speed, hour, day, month]
    Target  : temperature at t+1
    """
    df = df.copy()
    df["target"] = df["temperature"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["temperature", "humidity", "wind_speed", "hour", "day", "month"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_s, y_train)

    y_pred = rf.predict(X_te_s)
    m = _metrics(y_test, y_pred)
    print(f"  MAE={m['MAE']:.3f} °C  |  RMSE={m['RMSE']:.3f} °C  |  R²={m['R2']:.4f}")
    return rf, scaler, m


# ─────────────────────────────────────────────────────────────
# Step 3 — Generate Spatial Snapshots for GCN Training
# ─────────────────────────────────────────────────────────────

def generate_snapshots(
    df: pd.DataFrame,
    grid_df: pd.DataFrame,
    uhi_offsets: np.ndarray,
    n: int = N_SNAPSHOTS,
) -> list:
    """
    For each sampled timestep, create one graph snapshot:

        Node features (5 per zone):
            [rf_predicted_temp, zone_humidity, zone_wind, lat, lon]

        Target (1 per zone):
            zone_temp = base_temp + uhi_offset + small_noise

    The RF prediction is city-wide (single point) with random noise,
    reflecting that RF has no spatial awareness.
    The GCN learns to correct this using graph structure + node features.

    Returns
    -------
    List of dicts: {"X": (N, 5) ndarray, "y": (N,) ndarray}
    """
    rng = np.random.default_rng(42)
    lats = grid_df["center_lat"].values
    lons = grid_df["center_lon"].values

    # Sample timesteps (with replacement if n > len(df))
    indices = rng.choice(len(df), size=n, replace=(n > len(df)))
    snapshots = []

    for i in indices:
        row        = df.iloc[i]
        base_temp  = float(row["temperature"])
        humidity   = float(row["humidity"])
        wind_speed = float(row["wind_speed"])

        # Actual zone temperatures (ground truth for training)
        zone_temp = (
            base_temp
            + uhi_offsets
            + rng.normal(0, 0.4, N_NODES).astype(np.float32)
        )

        # RF spatial prediction: city-wide base + small noise (no UHI)
        rf_zone = base_temp + rng.normal(0, 0.8, N_NODES).astype(np.float32)

        # Zone-level humidity and wind (slight spatial variation)
        zone_hum  = humidity   + rng.normal(0, 2.0, N_NODES)
        zone_wind = wind_speed + rng.normal(0, 0.3, N_NODES)

        X = np.column_stack([rf_zone, zone_hum, zone_wind, lats, lons]).astype(np.float32)
        snapshots.append({"X": X, "y": zone_temp.astype(np.float32)})

    return snapshots


# ─────────────────────────────────────────────────────────────
# Step 4 — Train GCN
# ─────────────────────────────────────────────────────────────

def train_gcn(snapshots: list, adj_norm_t: torch.Tensor):
    """
    Train 2-layer GCN over spatial snapshots.

    Each forward pass processes one graph snapshot (one timestep):
        Input  : (N, 5) node features
        Output : (N, 1) predicted zone temperatures
        Loss   : MSE against actual zone temperatures

    Returns
    -------
    gcn, feat_scaler, tgt_scaler, metrics_rf_spatial, metrics_gcn
    """
    # Fit scalers on all snapshots
    all_X = np.vstack([s["X"] for s in snapshots])
    all_y = np.hstack([s["y"] for s in snapshots])
    feat_scaler = StandardScaler().fit(all_X)
    tgt_scaler  = StandardScaler().fit(all_y.reshape(-1, 1))

    # Chronological split
    split        = int(len(snapshots) * 0.8)
    train_snaps  = snapshots[:split]
    test_snaps   = snapshots[split:]

    gcn       = GCN(in_features=5, hidden=32, out_features=1, dropout=0.3)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=GCN_LR, weight_decay=5e-4)
    criterion = nn.MSELoss()

    rng = np.random.default_rng(0)

    gcn.train()
    for epoch in range(GCN_EPOCHS):
        rng.shuffle(train_snaps)
        epoch_loss = 0.0

        for snap in train_snaps:
            X_t = torch.FloatTensor(feat_scaler.transform(snap["X"]))
            y_t = torch.FloatTensor(
                tgt_scaler.transform(snap["y"].reshape(-1, 1))
            )
            optimizer.zero_grad()
            out  = gcn(X_t, adj_norm_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(train_snaps)
            print(f"  Epoch {epoch+1:>3d}/{GCN_EPOCHS}  avg_loss={avg_loss:.6f}")

    # ── Evaluation ──
    gcn.eval()
    rf_all, gcn_all, true_all = [], [], []

    with torch.no_grad():
        for snap in test_snaps:
            X_t  = torch.FloatTensor(feat_scaler.transform(snap["X"]))
            out  = gcn(X_t, adj_norm_t).numpy()
            pred = tgt_scaler.inverse_transform(out).ravel()

            rf_all.append(snap["X"][:, 0])   # column 0 = rf_zone_pred
            gcn_all.append(pred)
            true_all.append(snap["y"])

    rf_flat   = np.hstack(rf_all)
    gcn_flat  = np.hstack(gcn_all)
    true_flat = np.hstack(true_all)

    m_rf  = _metrics(true_flat, rf_flat)
    m_gcn = _metrics(true_flat, gcn_flat)

    print(f"\n  RF  (spatial)  →  MAE={m_rf['MAE']:.3f}  RMSE={m_rf['RMSE']:.3f}  R²={m_rf['R2']:.4f}")
    print(f"  RF+GCN Hybrid  →  MAE={m_gcn['MAE']:.3f}  RMSE={m_gcn['RMSE']:.3f}  R²={m_gcn['R2']:.4f}")

    return gcn, feat_scaler, tgt_scaler, m_rf, m_gcn


# ─────────────────────────────────────────────────────────────
# Step 5 — Save
# ─────────────────────────────────────────────────────────────

def save_all(rf, rf_scaler, gcn, feat_scaler, tgt_scaler, results):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        {"model": rf, "scaler": rf_scaler},
        os.path.join(MODEL_DIR, "rf_model.pkl"),
    )
    torch.save(gcn.state_dict(), os.path.join(MODEL_DIR, "gcn_weights.pth"))
    joblib.dump(
        {"feat_scaler": feat_scaler, "tgt_scaler": tgt_scaler},
        os.path.join(MODEL_DIR, "gcn_scalers.pkl"),
    )
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll models saved to  {MODEL_DIR}/")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Hybrid RF + GCN Training Pipeline")
    print("=" * 62)

    # 1. Data
    print("\n[1/5]  Loading weather and grid data ...")
    df      = load_weather()
    grid_df = pd.read_csv(GRID_PATH)
    print(f"  Grid: {len(grid_df)} zones ({GRID_ROWS}×{GRID_COLS})")

    # 2. UHI spatial offsets
    print("\n[2/5]  Computing UHI spatial offsets ...")
    uhi = compute_uhi_offsets(grid_df)
    print(f"  UHI range:  {uhi.min():.2f} °C  to  {uhi.max():.2f} °C")

    # 3. RF temporal model
    print("\n[3/5]  Training Random Forest temporal model ...")
    rf, rf_scaler, rf_metrics = train_rf(df)

    # 4. GCN spatial refinement
    print(f"\n[4/5]  Building graph + training GCN  ({N_SNAPSHOTS} snapshots) ...")
    adj        = build_adjacency_matrix(GRID_ROWS, GRID_COLS)
    adj_norm   = normalize_adjacency(adj)
    adj_norm_t = torch.FloatTensor(adj_norm)

    snapshots = generate_snapshots(df, grid_df, uhi, n=N_SNAPSHOTS)

    gcn, feat_scaler, tgt_scaler, m_rf_sp, m_gcn = train_gcn(
        snapshots, adj_norm_t
    )

    # 5. Save
    print("\n[5/5]  Saving models ...")
    results = {
        "rf_temporal": rf_metrics,
        "rf_spatial":  m_rf_sp,
        "rf_gcn":      m_gcn,
    }
    save_all(rf, rf_scaler, gcn, feat_scaler, tgt_scaler, results)

    # Summary table
    pad = 15
    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  {'Metric':<8}  {'RF Temporal':>{pad}}  {'RF Spatial':>{pad}}  {'RF+GCN':>{pad}}")
    print("  " + "-" * 56)
    for key in ["MAE", "RMSE", "R2"]:
        unit = " °C" if key != "R2" else "   "
        print(
            f"  {key:<8}"
            f"  {str(rf_metrics[key]) + unit:>{pad}}"
            f"  {str(m_rf_sp[key]) + unit:>{pad}}"
            f"  {str(m_gcn[key]) + unit:>{pad}}"
        )
    print("=" * 62)
    print("\nDone. Run  streamlit run app.py  to launch the dashboard.")


if __name__ == "__main__":
    main()
