import streamlit as st
import plotly.graph_objects as go
import requests
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.path import Path

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

@st.cache_data(ttl=5)
def fetch_data():
    try:
        r = requests.get(GOOGLE_SCRIPT_URL, timeout=10)
        return r.json().get("history", [])
    except:
        return []


def generate_clipped_surface(sensors, val_key):
    x, y, z = [], [], []
    for i, s in enumerate(sensors):
        val = s.get(val_key)
        if i in SENSOR_MAP and val is not None:
            x.append(SENSOR_MAP[i][0]);
            y.append(SENSOR_MAP[i][1]);
            z.append(float(val))

    if len(z) < 3: return None, None, None, 0, 0

    local_max = max(z)
    local_min = min(z)

    # 1. Create large grid for interpolation
    avg = np.mean(z)
    for px, py in [(-0.3, 1.5), (1.5, 1.5), (-0.3, 3.5), (1.5, 3.5)]:
        x.append(px);
        y.append(py);
        z.append(avg)

    grid_x, grid_y = np.meshgrid(np.linspace(-0.1, 1.3, 200), np.linspace(1.7, 3.3, 200))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # 2. Apply Clipping Mask
    poly_path = Path(OCTAGON_POLY)
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = poly_path.contains_points(points)
    mask = mask.reshape(grid_x.shape)
    grid_z[~mask] = np.nan

    return grid_x, grid_y, grid_z, local_max, local_min


def create_map(grid_z, title, colorscale, zmin, zmax):
    fig = go.Figure(data=go.Contour(
        z=grid_z, x=np.linspace(-0.1, 1.3, 200), y=np.linspace(1.7, 3.3, 200),
        colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=True,
        contours=dict(coloring='heatmap', showlabels=False),
        line=dict(width=0),
        connectgaps=False,
        colorbar=dict(thickness=15)
    ))

    # White Outline
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
# The Tortoise has returned!
st.title("🐢 rainford's queendom environmental monitor")
history = fetch_data()

if history:
    latest = history[-1]
    col1, col2 = st.columns(2)

    tgx, tgy, t_grid, t_max, t_min = generate_clipped_surface(latest["sensors"], "temp")
    hgx, hgy, h_grid, h_max, h_min = generate_clipped_surface(latest["sensors"], "hum")

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
    st.subheader("📈 Environmental Trends")

    df_list = []
    for entry in history:
        s_list = entry.get('sensors', [])
        t_vals = [s.get('temp') for s in s_list if s.get('temp') is not None]
        h_vals = [s.get('hum') for s in s_list if s.get('hum') is not None]

        df_list.append({
            "Time": entry.get('time', 'Unknown'),
            "Avg Temp": np.mean(t_vals) if t_vals else None,
            "Avg Hum": np.mean(h_vals) if h_vals else None
        })

    df = pd.DataFrame(df_list)

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=df["Time"], y=df["Avg Temp"], name="Temp (°C)",
                                   line=dict(color="#FF4B4B", width=3), yaxis="y1"))
    trend_fig.add_trace(go.Scatter(x=df["Time"], y=df["Avg Hum"], name="Humidity (%)",
                                   line=dict(color="#00D4FF", width=3), yaxis="y2"))

    trend_fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        height=400,
        margin=dict(l=50, r=50, t=20, b=20),
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

    if st.sidebar.button("🗑️ Reset Data"):
        requests.get(f"{GOOGLE_SCRIPT_URL}?action=clear")
        st.rerun()
else:
    st.info("Awaiting live data stream...")