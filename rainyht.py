import streamlit as st
import plotly.graph_objects as go
import requests
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.path import Path
from requests.exceptions import ReadTimeout

# ==========================================
# --- CONFIGURATION ---
# ==========================================
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbz2j4fbcorEgaOx4wt5h4D05jufu6AnY0OkUOqHfO_UFwCySQuASzBtDFOMkXu2L8xU/exec"

# COORDINATES: 4-5-6-7-3-2-1-0 clockwise
SENSOR_MAP = {
    4: (0.4, 1.8), 5: (0.0, 2.2), 6: (0.0, 2.8), 7: (0.4, 3.2),
    3: (0.8, 3.2), 2: (1.2, 2.8), 1: (1.2, 2.2), 0: (0.8, 1.8)
}

# The Cookie Cutter Shape (Octagon)
OCTAGON_POLY = [
    (0.4, 1.8), (0.0, 2.2), (0.0, 2.8), (0.4, 3.2),
    (0.8, 3.2), (1.2, 2.8), (1.2, 2.2), (0.8, 1.8),
    (0.4, 1.8)  # Close the loop
]

st.set_page_config(page_title="Rainford's Queendom", layout="wide")


# ==========================================
# --- DATA PROCESSING ---
# ==========================================

@st.cache_data(ttl=60)
def fetch_and_clean_data():
    try:
        # TIMEOUT 60 SECONDS
        r = requests.get(GOOGLE_SCRIPT_URL, timeout=60)
        r.raise_for_status()
        raw_history = r.json().get("history", [])

        if not raw_history:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(raw_history)

        # Force UTC-aware datetime conversion
        df['time'] = pd.to_datetime(df['time'], utc=True)

        # Helper to extract averages from the sensor list column
        def calculate_avg(sensor_list, key):
            if not isinstance(sensor_list, list): return np.nan
            vals = [s.get(key) for s in sensor_list if s.get(key) is not None]
            return np.mean(vals) if vals else np.nan

        df['avg_temp'] = df['sensors'].apply(lambda x: calculate_avg(x, 'temp'))
        df['avg_hum'] = df['sensors'].apply(lambda x: calculate_avg(x, 'hum'))

        return df

    except ReadTimeout:
        st.error("⚠️ Timeout: Google Sheet took too long to respond. Try deleting old rows in the sheet.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def generate_clipped_surface(sensors, val_key):
    x, y, z = [], [], []
    for i, s in enumerate(sensors):
        val = s.get(val_key)
        if i in SENSOR_MAP and val is not None:
            x.append(SENSOR_MAP[i][0])
            y.append(SENSOR_MAP[i][1])
            z.append(float(val))

    if len(z) < 3: return None, None, None, 0, 0

    local_max, local_min = max(z), min(z)
    avg = np.mean(z)

    # Padding points to help interpolation reach edges
    x_ext = x + [-0.3, 1.5, -0.3, 1.5]
    y_ext = y + [1.5, 1.5, 3.5, 3.5]
    z_ext = z + [avg, avg, avg, avg]

    # Coarser grid (100x100) for speed
    grid_x, grid_y = np.meshgrid(np.linspace(-0.1, 1.3, 100), np.linspace(1.7, 3.3, 100))
    grid_z = griddata((x_ext, y_ext), z_ext, (grid_x, grid_y), method='cubic')

    # Clipping to Octagon
    poly_path = Path(OCTAGON_POLY)
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = poly_path.contains_points(points).reshape(grid_x.shape)
    grid_z[~mask] = np.nan

    return grid_x, grid_y, grid_z, local_max, local_min


def create_map(grid_z, title, colorscale, zmin, zmax):
    fig = go.Figure(data=go.Contour(
        z=grid_z, x=np.linspace(-0.1, 1.3, 100), y=np.linspace(1.7, 3.3, 100),
        colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
        contours=dict(coloring='heatmap', showlabels=False),
        line=dict(width=0), connectgaps=False,
        colorbar=dict(thickness=15)
    ))

    path_str = "M 0.4 1.8 L 0.0 2.2 L 0.0 2.8 L 0.4 3.2 L 0.8 3.2 L 1.2 2.8 L 1.2 2.2 L 0.8 1.8 Z"
    fig.add_shape(type="path", path=path_str, line=dict(color="white", width=3))

    fig.update_layout(title=title, height=500, template="plotly_dark",
                      xaxis=dict(visible=False, range=[-0.1, 1.3]),
                      yaxis=dict(visible=False, scaleanchor="x", range=[1.7, 3.3]),
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ==========================================
# --- UI ---
# ==========================================
st.title("🐢 rainford's queendom environmental monitor")

df = fetch_and_clean_data()

if not df.empty:
    # --- Filter Logic (Hardcoded 48 Hours) ---
    lookback_hours = 48
    now_utc = pd.Timestamp.now(tz='UTC')
    cutoff_time = now_utc - pd.Timedelta(hours=lookback_hours)
    
    filtered_df = df[df['time'] >= cutoff_time]

    # --- Live Maps (Latest Point Only) ---
    latest_row = df.iloc[-1]
    col1, col2 = st.columns(2)

    tgx, tgy, t_grid, t_max, t_min = generate_clipped_surface(latest_row["sensors"], "temp")
    hgx, hgy, h_grid, h_max, h_min = generate_clipped_surface(latest_row["sensors"], "hum")

    with col1:
        if t_grid is not None:
            st.plotly_chart(create_map(t_grid, f"Current Temp (Peak: {t_max}°C)", "Hot", t_min, t_max),
                            use_container_width=True)

    with col2:
        if h_grid is not None:
            st.plotly_chart(create_map(h_grid, f"Current Humidity (Peak: {h_max}%)", "Teal", h_min, h_max),
                            use_container_width=True)

    # --- Historical Trend ---
    st.markdown("---")
    st.subheader(f"📈 Environmental Trends (Last {lookback_hours} Hours)")

    if not filtered_df.empty:
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(x=filtered_df["time"], y=filtered_df["avg_temp"], name="Temp (°C)",
                                       line=dict(color="#FF4B4B", width=2), yaxis="y1"))
        trend_fig.add_trace(go.Scatter(x=filtered_df["time"], y=filtered_df["avg_hum"], name="Humidity (%)",
                                       line=dict(color="#00D4FF", width=2), yaxis="y2"))

        # FORCE X-AXIS TO SHOW FULL 48 HOURS REGARDLESS OF DATA
        trend_fig.update_layout(
            template="plotly_dark", hovermode="x unified", height=400,
            xaxis=dict(
                range=[cutoff_time, now_utc],
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text="Temperature (°C)", font=dict(color="#FF4B4B")),
                tickfont=dict(color="#FF4B4B")
            ),
            yaxis2=dict(
                title=dict(text="Humidity (%)", font=dict(color="#00D4FF")),
                tickfont=dict(color="#00D4FF"),
                overlaying="y", side="right"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.warning(f"No data found in the last {lookback_hours} hours.")

    # --- Admin Controls ---
    with st.expander("⚙️ Admin Controls"):
        if st.button("🗑️ Reset Data", type="primary"):
            requests.get(f"{GOOGLE_SCRIPT_URL}?action=clear")
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Awaiting live data stream...")
