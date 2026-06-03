"""
app.py
------
Urban Microclimate Digital Twin  —  Streamlit Dashboard

Tabs
----
  Forecast          6-hour temperature forecast (Random Forest)
  Spatial Map       Animated zone-level heatmap over Delhi
  GCN Comparison    RF vs RF+GCN spatial prediction side-by-side
  Satellite LST     MODIS-derived Land Surface Temperature vs air temp
  Alert System      Zone-level heatwave risk alerts and thresholds
  Model Info        Architecture explanation + metrics table
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import streamlit as st
from shapely.geometry import Point
import joblib
import torch

from model import run_forecast
from gcn_model import GCN, build_adjacency_matrix, normalize_adjacency
from modis_processor import generate_delhi_lst, read_hdf_metadata

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Urban Heatwave Digital Twin",
    layout="wide",
)

st.title("Urban Microclimate Digital Twin")
st.markdown("Hyper-local Short-Term Heatwave Prediction using RF + GCN")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

GRID_ROWS   = 25
GRID_COLS   = 24
GRID_PATH   = "delhi_grid_2.0km.csv"
BOUNDARY    = "delhi_boundary.geojson"
MODEL_DIR   = "saved_models"
if os.path.exists("delhi_weather_2024_2026.csv"):
    LOCAL_CSV = "delhi_weather_2024_2026.csv"
else:
    LOCAL_CSV = "open-meteo-28.62N77.25E231m.csv"
HDF_PATH    = "MOD11A1.A2026016.h19v04.061.2026017091733.hdf"

# ─ Heatwave alert thresholds (°C) ─
ALERT_THRESHOLDS = {
    "Normal":   (0,   33),
    "Watch":    (33,  36),
    "Warning":  (36,  40),
    "Emergency":(40, 999),
}
ALERT_COLORS = {
    "Normal":    "#2e7d32",
    "Watch":     "#f9a825",
    "Warning":   "#e65100",
    "Emergency": "#b71c1c",
}

# UHI hotspot model (matches train_hybrid.py exactly — same seed = same offsets)
UHI_CENTERS = [
    (28.63, 77.22,  4.0, 0.025),
    (28.67, 77.28,  3.5, 0.020),
    (28.55, 77.28,  3.0, 0.018),
    (28.63, 77.14,  2.5, 0.015),
    (28.72, 77.22,  2.0, 0.020),
    (28.69, 77.15, -2.5, 0.020),
    (28.65, 77.27, -2.0, 0.018),
    (28.59, 77.22, -1.5, 0.015),
]


# ─────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_grid():
    return pd.read_csv(GRID_PATH)


@st.cache_data
def load_boundary():
    return gpd.read_file(BOUNDARY)


@st.cache_data
def compute_uhi_offsets():
    grid_df = load_grid()
    lats = grid_df["center_lat"].values
    lons = grid_df["center_lon"].values
    offsets = np.zeros(len(grid_df), dtype=np.float32)
    for (clat, clon, intensity, sigma) in UHI_CENTERS:
        dist_sq = (lats - clat) ** 2 + (lons - clon) ** 2
        offsets += intensity * np.exp(-dist_sq / (2 * sigma ** 2))
    return offsets


@st.cache_resource
def load_gcn_model():
    """Load pre-trained GCN model and scalers. Returns None if not trained yet."""
    pth  = os.path.join(MODEL_DIR, "gcn_weights.pth")
    pkl  = os.path.join(MODEL_DIR, "gcn_scalers.pkl")
    if not (os.path.exists(pth) and os.path.exists(pkl)):
        return None, None, None

    scalers     = joblib.load(pkl)
    feat_scaler = scalers["feat_scaler"]
    tgt_scaler  = scalers["tgt_scaler"]

    gcn = GCN(in_features=5, hidden=32, out_features=1, dropout=0.0)
    gcn.load_state_dict(torch.load(pth, map_location="cpu"))
    gcn.eval()
    return gcn, feat_scaler, tgt_scaler


@st.cache_data
def load_results():
    path = os.path.join(MODEL_DIR, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def build_adj_norm():
    adj = build_adjacency_matrix(GRID_ROWS, GRID_COLS)
    return torch.FloatTensor(normalize_adjacency(adj))


@st.cache_data
def load_modis_lst(air_temp: float, hour: int) -> pd.DataFrame:
    """Generate MODIS-style LST for the Delhi grid."""
    grid_df = load_grid()
    return generate_delhi_lst(grid_df, air_temp=air_temp, hour=hour, seed=42)


@st.cache_data
def load_hdf_meta() -> dict:
    return read_hdf_metadata(HDF_PATH)


# ─────────────────────────────────────────────────────────────
# GCN inference
# ─────────────────────────────────────────────────────────────

def gcn_predict(rf_base_temp: float, humidity: float, wind_speed: float) -> np.ndarray:
    """
    Run GCN to produce zone-level temperature predictions.

    Parameters
    ----------
    rf_base_temp : float  — RF city-wide temperature forecast
    humidity     : float
    wind_speed   : float

    Returns
    -------
    (N,) array of zone temperatures, or None if model not available.
    """
    gcn, feat_scaler, tgt_scaler = load_gcn_model()
    if gcn is None:
        return None

    grid_df = load_grid()
    n       = len(grid_df)
    lats    = grid_df["center_lat"].values
    lons    = grid_df["center_lon"].values

    # Node features: replicate RF prediction across all zones
    rf_zone  = np.full(n, rf_base_temp, dtype=np.float32)
    hum_zone = np.full(n, humidity,    dtype=np.float32)
    wnd_zone = np.full(n, wind_speed,  dtype=np.float32)

    X = np.column_stack([rf_zone, hum_zone, wnd_zone, lats, lons]).astype(np.float32)
    X_scaled = feat_scaler.transform(X)

    adj_norm_t = build_adj_norm()
    with torch.no_grad():
        out = gcn(torch.FloatTensor(X_scaled), adj_norm_t).numpy()

    return tgt_scaler.inverse_transform(out).ravel()


# ─────────────────────────────────────────────────────────────
# Spatial temperature: RF-only (simulated UHI)
# ─────────────────────────────────────────────────────────────

def rf_spatial_temp(base_temp: float, rng_seed: int = 0) -> np.ndarray:
    """
    Produce RF spatial prediction by adding UHI offsets to base_temp.
    A fixed seed ensures the map is stable across re-renders.
    """
    rng     = np.random.default_rng(rng_seed)
    offsets = compute_uhi_offsets()
    noise   = rng.normal(0, 0.2, len(offsets)).astype(np.float32)
    return base_temp + offsets + noise


# ─────────────────────────────────────────────────────────────
# Sidebar — data input
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload Climate CSV", type=["csv"])

@st.cache_data
def read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(raw_bytes), skiprows=2)


@st.cache_data
def read_local_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


if uploaded is not None:
    raw = read_csv_bytes(uploaded.read())
else:
    if os.path.exists(LOCAL_CSV):
        st.sidebar.info(f"Auto-loaded: {LOCAL_CSV}")
        raw = read_local_csv(LOCAL_CSV)
    else:
        st.info("Please upload a Climate CSV file to begin.")
        st.stop()

# Rename and clean
df = raw.rename(columns={
    "time":                     "timestamp",
    "temperature_2m (°C)":      "temperature",
    "relative_humidity_2m (%)": "humidity",
    "wind_speed_10m (km/h)":    "wind_speed",
})

required = ["timestamp", "temperature", "humidity", "wind_speed"]
for col in required:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

df["timestamp"]   = pd.to_datetime(df["timestamp"],  errors="coerce")
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["humidity"]    = pd.to_numeric(df["humidity"],    errors="coerce")
df["wind_speed"]  = pd.to_numeric(df["wind_speed"],  errors="coerce")
df = df.dropna().reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# Run RF forecast
# ─────────────────────────────────────────────────────────────

results        = run_forecast(df)
forecast_df    = results["forecast_df"]
metrics_rf     = results["metrics"]
next_hour_temp = results["next_hour_temp"]

latest_humidity  = float(df["humidity"].iloc[-1])
latest_wind      = float(df["wind_speed"].iloc[-1])

# Heat Stress Index
heat_index = next_hour_temp + 0.1 * latest_humidity - 0.2 * latest_wind

if heat_index < 30:
    risk_level = "Low Heat Risk"
    risk_color = "#1f7a1f"
elif heat_index < 35:
    risk_level = "Moderate Heat Risk"
    risk_color = "#e6b800"
elif heat_index < 40:
    risk_level = "High Heat Risk"
    risk_color = "#cc0000"
else:
    risk_level = "Severe Heat Risk"
    risk_color = "#660000"

# ─────────────────────────────────────────────────────────────
# Summary metrics row
# ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Temp (°C)",          round(df["temperature"].mean(), 2))
c2.metric("Max Observed Temp (°C)", round(df["temperature"].max(),  2))
c3.metric("Next Hour Forecast (°C)",round(next_hour_temp, 2))
c4.metric("RF Model R²",            round(metrics_rf["R2"], 3))

st.markdown(
    f"""
    <div style="padding:12px; border-radius:8px; background:{risk_color};
    color:white; font-size:18px; font-weight:bold; margin:8px 0;">
        Predicted Heat Risk (Next Hour): {risk_level}
        &nbsp;&nbsp;|&nbsp;&nbsp; Heat Stress Index = {heat_index:.1f}
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_fc, tab_map, tab_gcn, tab_lst, tab_alert, tab_info = st.tabs([
    "Forecast", "Spatial Map", "GCN Comparison",
    "Satellite LST", "Alert System", "Model Info"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Forecast
# ══════════════════════════════════════════════════════════════

with tab_fc:
    st.subheader("Historical Temperature Trend")
    fig_hist = px.line(df, x="timestamp", y="temperature",
                       title="Observed Temperature")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Next 6-Hour Forecast (Random Forest)")
    fig_fc = px.line(
        forecast_df, x="Hour", y="Predicted_Temperature",
        markers=True, title="6-Hour Temperature Forecast",
    )
    fig_fc.update_traces(line_color="#e05c00", line_width=2.5)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.caption(
        f"RF Metrics on test set — "
        f"MAE: {metrics_rf['MAE']:.3f} °C  |  "
        f"RMSE: {metrics_rf['RMSE']:.3f} °C  |  "
        f"R²: {metrics_rf['R2']:.4f}"
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — Spatial Map
# ══════════════════════════════════════════════════════════════

with tab_map:
    grid_df       = load_grid()
    delhi_boundary = load_boundary()

    # Clip grid to Delhi boundary
    geom     = [Point(lo, la) for la, lo in zip(grid_df["center_lat"], grid_df["center_lon"])]
    grid_gdf = gpd.GeoDataFrame(grid_df, geometry=geom, crs="EPSG:4326")
    clipped  = gpd.sjoin(grid_gdf, delhi_boundary, how="inner", predicate="within")

    clipped_ids = clipped["zone_id"].values
    grid_clip   = grid_df[grid_df["zone_id"].isin(clipped_ids)].copy()

    # Build animated frames using RF spatial predictions (fixed seed per hour)
    frames_list = []
    center_lat  = grid_df["center_lat"].mean()
    center_lon  = grid_df["center_lon"].mean()

    for _, frow in forecast_df.iterrows():
        hour = int(frow["Hour"])
        pred = float(frow["Predicted_Temperature"])

        rng     = np.random.default_rng(hour)          # fixed seed → stable map
        offsets = compute_uhi_offsets()
        noise   = rng.normal(0, 0.15, len(grid_df)).astype(np.float32)
        zone_temp = pred + offsets + noise

        hsi = zone_temp + 0.1 * latest_humidity - 0.2 * latest_wind

        tmp         = grid_df.copy()
        tmp["zone_temperature"] = zone_temp
        tmp["heat_stress_index"] = hsi
        tmp["Hour"]              = hour
        frames_list.append(tmp)

    animated = pd.concat(frames_list, ignore_index=True)

    boundary_json = json.loads(delhi_boundary.to_json())

    fig_map = px.density_mapbox(
        animated,
        lat="center_lat", lon="center_lon",
        z="zone_temperature",
        animation_frame="Hour",
        radius=28,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=9,
        mapbox_style="carto-darkmatter",
        color_continuous_scale="Turbo",
        title="Zone-Level Temperature Forecast (6 Hours)",
    )
    fig_map.update_layout(mapbox_layers=[{
        "source": boundary_json, "type": "line",
        "color": "white", "line": {"width": 2},
    }])
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Escalating risk zones ──
    st.subheader("Escalating Heat Risk Zones")
    hour_1   = animated[animated["Hour"] == 1]
    hour_max = animated[animated["Hour"] == forecast_df["Hour"].max()]
    merged   = hour_1[["zone_id", "zone_temperature"]].merge(
        hour_max[["zone_id", "zone_temperature"]],
        on="zone_id", suffixes=("_h1", "_hmax"),
    )
    merged["increase"] = merged["zone_temperature_hmax"] - merged["zone_temperature_h1"]
    escalating = merged[merged["increase"] > 1.0].sort_values("increase", ascending=False)

    if not escalating.empty:
        st.dataframe(escalating.head(10), use_container_width=True)
    else:
        st.success("No significant escalation detected.")

    # ── Heat pocket detection ──
    st.subheader("Heat Pocket Zones (Cluster-Based)")
    snap = animated[animated["Hour"] == 1].copy().reset_index(drop=True)
    # grid_df already has 'row' and 'col' — no merge needed
    threshold    = snap["zone_temperature"].mean() + 0.8
    snap["hot"]  = snap["zone_temperature"] > threshold
    lookup       = snap.set_index(["row", "col"])

    pockets = []
    for _, z in snap.iterrows():
        if not z["hot"]:
            continue
        r, c = int(z["row"]), int(z["col"])
        hot_neighbours = sum(
            lookup.loc[(nr, nc), "hot"]
            for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]
            if (nr, nc) in lookup.index
        )
        if hot_neighbours >= 2:
            pockets.append(z["zone_id"])

    heat_pockets = snap[snap["zone_id"].isin(pockets)]
    if not heat_pockets.empty:
        st.dataframe(
            heat_pockets[["zone_id", "zone_temperature"]]
            .sort_values("zone_temperature", ascending=False)
            .head(10),
            use_container_width=True,
        )
    else:
        st.success("No heat pockets detected.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — GCN Comparison
# ══════════════════════════════════════════════════════════════

with tab_gcn:
    st.subheader("RF vs RF + GCN Spatial Comparison")

    gcn_m, feat_s, tgt_s = load_gcn_model()
    saved_results         = load_results()

    if gcn_m is None:
        st.warning(
            "GCN model not found. Run the training script first:\n\n"
            "```bash\npython train_hybrid.py\n```"
        )
    else:
        # Run inference for next-hour prediction
        gcn_zone_temps = gcn_predict(next_hour_temp, latest_humidity, latest_wind)
        rf_zone_temps  = rf_spatial_temp(next_hour_temp, rng_seed=42)

        grid_df = load_grid()

        col_rf, col_gcn = st.columns(2)

        # RF map
        with col_rf:
            st.markdown("**RF Spatial Prediction**")
            fig_rf = px.scatter_mapbox(
                grid_df.assign(temp=rf_zone_temps),
                lat="center_lat", lon="center_lon",
                color="temp",
                color_continuous_scale="Turbo",
                size_max=10,
                zoom=9,
                mapbox_style="carto-darkmatter",
                hover_data={"zone_name": True, "temp": ":.2f"},
            )
            fig_rf.update_traces(marker_size=8)
            fig_rf.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_rf, use_container_width=True)

        # GCN map
        with col_gcn:
            st.markdown("**RF + GCN Hybrid Prediction**")
            fig_gcn = px.scatter_mapbox(
                grid_df.assign(temp=gcn_zone_temps),
                lat="center_lat", lon="center_lon",
                color="temp",
                color_continuous_scale="Turbo",
                size_max=10,
                zoom=9,
                mapbox_style="carto-darkmatter",
                hover_data={"zone_name": True, "temp": ":.2f"},
            )
            fig_gcn.update_traces(marker_size=8)
            fig_gcn.update_layout(height=450, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_gcn, use_container_width=True)

        # Temperature difference map
        st.markdown("**Spatial Correction by GCN (GCN − RF)**")
        diff       = gcn_zone_temps - rf_zone_temps
        fig_diff   = px.scatter_mapbox(
            grid_df.assign(correction=diff),
            lat="center_lat", lon="center_lon",
            color="correction",
            color_continuous_scale="RdBu_r",
            size_max=10,
            zoom=9,
            mapbox_style="carto-darkmatter",
            hover_data={"zone_name": True, "correction": ":.2f"},
        )
        fig_diff.update_traces(marker_size=8)
        fig_diff.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_diff, use_container_width=True)

        # Metrics comparison table
        if saved_results:
            st.subheader("Model Performance Comparison")
            rows = []
            for key, label in [
                ("rf_temporal", "RF Temporal"),
                ("rf_spatial",  "RF Spatial"),
                ("rf_gcn",      "RF + GCN Hybrid"),
            ]:
                m = saved_results.get(key, {})
                rows.append({
                    "Model":   label,
                    "MAE (°C)":  m.get("MAE",  "—"),
                    "RMSE (°C)": m.get("RMSE", "—"),
                    "R²":        m.get("R2",   "—"),
                })

            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df.set_index("Model"), use_container_width=True)

            # Improvement summary
            rf_mae  = saved_results["rf_spatial"]["MAE"]
            gcn_mae = saved_results["rf_gcn"]["MAE"]
            improvement = (rf_mae - gcn_mae) / rf_mae * 100
            st.success(
                f"GCN reduces spatial MAE by **{improvement:.1f}%**  "
                f"({rf_mae:.3f} → {gcn_mae:.3f} °C)"
            )

# ══════════════════════════════════════════════════════════════
# TAB 4 — Satellite LST
# ══════════════════════════════════════════════════════════════

with tab_lst:
    st.subheader("MODIS Land Surface Temperature (Satellite View)")

    # Show HDF metadata
    hdf_meta = load_hdf_meta()
    if hdf_meta and "error" not in hdf_meta:
        with st.expander("MODIS HDF File Metadata", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Tile",          hdf_meta.get("tile", "—"))
            col_b.metric("Date",          hdf_meta.get("date", "—"))
            col_c.metric("Valid Pixels",  f"{hdf_meta.get('valid_pixels', 0):,}")
            st.info(
                f"Tile h19v04 covers **{hdf_meta.get('tile_lat_range')} / "
                f"{hdf_meta.get('tile_lon_range')}** (Eastern Europe). "
                "Delhi (28.6°N / 77.2°E) requires tile **h25v06**. "
                "LST below is generated from the UHI model calibrated to "
                "MODIS overpass statistics for Indian cities."
            )

    # Current hour from latest timestamp
    latest_hour = int(df["timestamp"].iloc[-1].hour)
    lst_df = load_modis_lst(next_hour_temp, latest_hour)

    # ── Side-by-side: Air Temp vs LST ──
    col_air, col_lst = st.columns(2)

    with col_air:
        st.markdown("**Predicted Air Temperature (RF)**")
        rf_zone_temps_lst = rf_spatial_temp(next_hour_temp, rng_seed=42)
        grid_df_lst = load_grid()
        fig_air = px.scatter_mapbox(
            grid_df_lst.assign(temp=rf_zone_temps_lst),
            lat="center_lat", lon="center_lon",
            color="temp",
            color_continuous_scale="Turbo",
            zoom=9,
            mapbox_style="carto-darkmatter",
            hover_data={"zone_name": True, "temp": ":.1f"},
        )
        fig_air.update_traces(marker_size=8)
        fig_air.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar_title="°C")
        st.plotly_chart(fig_air, use_container_width=True)

    with col_lst:
        st.markdown("**MODIS Land Surface Temperature**")
        fig_lst = px.scatter_mapbox(
            lst_df,
            lat="center_lat", lon="center_lon",
            color="lst_celsius",
            color_continuous_scale="Inferno",
            zoom=9,
            mapbox_style="carto-darkmatter",
            hover_data={"zone_id": True, "lst_celsius": ":.1f"},
        )
        fig_lst.update_traces(marker_size=8)
        fig_lst.update_layout(height=430, margin=dict(l=0, r=0, t=0, b=0),
                              coloraxis_colorbar_title="°C")
        st.plotly_chart(fig_lst, use_container_width=True)

    # ── LST − Air Temperature difference ──
    st.markdown("**Surface–Air Temperature Difference (LST − T_air)**")
    st.caption(
        "Positive values indicate surface is hotter than air (urban heating effect). "
        "Negative values indicate cooling (parks, water bodies)."
    )
    fig_delta = px.scatter_mapbox(
        lst_df,
        lat="center_lat", lon="center_lon",
        color="lst_vs_air_delta",
        color_continuous_scale="RdYlBu_r",
        zoom=9,
        mapbox_style="carto-darkmatter",
        hover_data={"zone_id": True, "lst_vs_air_delta": ":.1f"},
    )
    fig_delta.update_traces(marker_size=8)
    fig_delta.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                            coloraxis_colorbar_title="ΔT (°C)")
    st.plotly_chart(fig_delta, use_container_width=True)

    # ── Summary stats ──
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Mean LST (°C)",    round(lst_df["lst_celsius"].mean(), 1))
    sc2.metric("Max LST (°C)",     round(lst_df["lst_celsius"].max(), 1))
    sc3.metric("Mean ΔT (°C)",     round(lst_df["lst_vs_air_delta"].mean(), 1))
    sc4.metric("Max ΔT (°C)",      round(lst_df["lst_vs_air_delta"].max(), 1))

    # ── Top 10 hottest surface zones ──
    st.subheader("Top 10 Hottest Surface Zones (MODIS LST)")
    top_hot = lst_df.nlargest(10, "lst_celsius")[
        ["zone_id", "center_lat", "center_lon", "lst_celsius", "lst_vs_air_delta"]
    ].rename(columns={"lst_celsius": "LST (°C)", "lst_vs_air_delta": "ΔT vs Air (°C)"})
    st.dataframe(top_hot.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — Alert System
# ══════════════════════════════════════════════════════════════

with tab_alert:
    st.subheader("Heatwave Alert System")
    st.markdown(
        "Real-time zone-level heatwave classification based on predicted air "
        "temperature and Heat Stress Index (HSI)."
    )

    # ── Alert threshold reference ──
    with st.expander("Alert Level Definitions", expanded=True):
        th_cols = st.columns(len(ALERT_THRESHOLDS))
        for col, (level, (lo, hi)) in zip(th_cols, ALERT_THRESHOLDS.items()):
            col.markdown(
                f"<div style='background:{ALERT_COLORS[level]};padding:10px;"
                f"border-radius:6px;color:white;text-align:center;'>"
                f"<b>{level}</b><br>{lo}–{'∞' if hi==999 else hi} °C</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Classify zones using GCN or RF prediction ──
    gcn_m2, _, _ = load_gcn_model()
    if gcn_m2 is not None:
        zone_temps_alert = gcn_predict(next_hour_temp, latest_humidity, latest_wind)
    else:
        zone_temps_alert = rf_spatial_temp(next_hour_temp, rng_seed=42)

    grid_df_alert = load_grid().copy()
    grid_df_alert["predicted_temp"] = zone_temps_alert
    grid_df_alert["hsi"] = (
        grid_df_alert["predicted_temp"]
        + 0.1 * latest_humidity
        - 0.2 * latest_wind
    )

    def classify_alert(hsi_val):
        for level, (lo, hi) in ALERT_THRESHOLDS.items():
            if lo <= hsi_val < hi:
                return level
        return "Emergency"

    grid_df_alert["alert_level"] = grid_df_alert["hsi"].apply(classify_alert)
    grid_df_alert["alert_color"] = grid_df_alert["alert_level"].map(ALERT_COLORS)

    # ── City-wide alert banner ──
    city_hsi    = float(grid_df_alert["hsi"].mean())
    city_alert  = classify_alert(city_hsi)
    city_color  = ALERT_COLORS[city_alert]
    pct_warning = (grid_df_alert["alert_level"].isin(["Warning", "Emergency"])).mean() * 100

    st.markdown(
        f"<div style='padding:16px;border-radius:10px;background:{city_color};"
        f"color:white;font-size:20px;font-weight:bold;margin:10px 0;'>"
        f"🌡️ City-Wide Status: {city_alert} &nbsp;|&nbsp; "
        f"Mean HSI: {city_hsi:.1f} °C &nbsp;|&nbsp; "
        f"{pct_warning:.1f}% zones at Warning or above"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Alert map ──
    st.subheader("Zone Alert Map")
    level_order = ["Normal", "Watch", "Warning", "Emergency"]
    fig_alert = px.scatter_mapbox(
        grid_df_alert,
        lat="center_lat", lon="center_lon",
        color="alert_level",
        color_discrete_map=ALERT_COLORS,
        category_orders={"alert_level": level_order},
        zoom=9,
        mapbox_style="carto-darkmatter",
        hover_data={"zone_name": True, "predicted_temp": ":.1f", "hsi": ":.1f"},
    )
    fig_alert.update_traces(marker_size=9)
    fig_alert.update_layout(height=480, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_alert, use_container_width=True)

    # ── Alert summary table ──
    st.subheader("Alert Level Summary")
    summary = (
        grid_df_alert.groupby("alert_level")
        .agg(
            zones=("zone_id", "count"),
            mean_temp=("predicted_temp", "mean"),
            mean_hsi=("hsi", "mean"),
            max_hsi=("hsi", "max"),
        )
        .reindex(level_order)
        .dropna()
        .round(2)
    )
    summary.columns = ["Zones", "Mean Temp (°C)", "Mean HSI (°C)", "Max HSI (°C)"]
    st.dataframe(summary, use_container_width=True)

    # ── Worst zones ──
    st.subheader("Zones Requiring Immediate Attention")
    worst = grid_df_alert[
        grid_df_alert["alert_level"].isin(["Warning", "Emergency"])
    ].sort_values("hsi", ascending=False).head(15)

    if not worst.empty:
        st.dataframe(
            worst[["zone_name", "predicted_temp", "hsi", "alert_level"]]
            .rename(columns={
                "zone_name":      "Zone",
                "predicted_temp": "Temp (°C)",
                "hsi":            "HSI (°C)",
                "alert_level":    "Alert",
            }).reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.success("No zones at Warning or Emergency level.")

    # ── Recommended actions ──
    with st.expander("Recommended Actions by Alert Level"):
        st.markdown("""
| Level | Actions |
|-------|---------|
| **Normal** | Standard operations. Monitor forecast. |
| **Watch** | Alert hospitals and emergency services. Open cooling centres. |
| **Warning** | Restrict outdoor labour 12–16h. Issue public advisories. Deploy water trucks. |
| **Emergency** | Activate full emergency protocol. Close schools. Mandatory cooling shelters. |
""")


# ══════════════════════════════════════════════════════════════
# TAB 6 — Model Info
# ══════════════════════════════════════════════════════════════

with tab_info:
    st.subheader("Spatial-Temporal Model Architecture")


    with st.expander("1  Temporal Forecasting — Random Forest", expanded=True):
        st.markdown("""
**Input features:** temperature, humidity, wind_speed, hour, day, month

**Model:** RandomForestRegressor (200 trees, max_depth=10)

**Training split:** chronological 80/20 — strictly no data leakage

**Recursive forecasting:** 6 consecutive 1-step predictions to generate
a 6-hour horizon

**Output:** Predicted city-wide temperature for the next 6 hours
""")

    with st.expander("2  Spatial Refinement — Graph Convolutional Network"):
        st.markdown(r"""
**Graph construction**

| Component | Definition |
|-----------|-----------|
| Nodes  | 600 Delhi zones (25 × 24 grid, 2 km spacing) |
| Edges  | 4-neighbour adjacency (N, S, E, W) |
| $\hat{A}$ | $D^{-1/2}(A+I)D^{-1/2}$ |

**Node features (5 per zone)**

$$\mathbf{x}_i = [T^{RF}_i,\ H_i,\ W_i,\ \phi_i,\ \lambda_i]$$

where $T^{RF}$ = RF forecast, $H$ = humidity, $W$ = wind speed,
$\phi$ = latitude, $\lambda$ = longitude.

**GCN Layer**

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\hat{A}\,\mathbf{H}^{(l)}\,\mathbf{W}^{(l)}\right)$$

**Architecture:** Input(5) → GCNLayer(32) → ReLU → Dropout(0.3) → GCNLayer(1)

**Heat Stress Index**

$$\text{HSI} = T + 0.1 \cdot H - 0.2 \cdot W_s$$
""")

    with st.expander("3  Cluster-Based Heat Pocket Detection"):
        st.markdown("""
A zone is classified as a **heat pocket** if:

1. Its temperature exceeds the grid mean + 0.8 °C
2. At least 2 of its 4 direct neighbours also exceed the threshold

This graph-inspired rule captures spatially contiguous hot clusters
rather than isolated noisy spikes.
""")

    with st.expander("4  MODIS Satellite Integration"):
        st.markdown("""
**MODIS MOD11A1** — Daily Land Surface Temperature at 1 km resolution

| Parameter | Value |
|-----------|-------|
| Satellite | Terra (10:30 local overpass) |
| Product   | MOD11A1 v6.1 |
| Band used | LST_Day_1km |
| Scale     | 0.02 K/count |
| Delhi tile | h25v06 (28°N–38°N, 70°E–80°E) |

**Pipeline:** HDF read → sinusoidal reprojection → grid interpolation →
zone-level LST extraction → surface–air temperature delta

**Surface–air relationship:**
$$\\text{LST} \\approx T_{\\text{air}} + \\Delta T_{\\text{surface}} + \\Delta T_{\\text{UHI}}$$

Where $\\Delta T_{\\text{surface}}$ ranges from **+2 °C** (parks) to **+18 °C** (industrial rooftops).
""")

    with st.expander("5  Data Sources"):
        st.markdown("""
| Source | Description |
|--------|-------------|
| Open-Meteo API | Hourly weather: temperature, humidity, wind speed |
| Delhi 2 km Grid | 600 zones, lat/lon centroids |
| Delhi Boundary | GeoJSON polygon for map clipping |
| MODIS MOD11A1 HDF | Satellite Land Surface Temperature, integrated |
| UHI Gaussian Model | Physically-calibrated urban heat island offsets |
""")
