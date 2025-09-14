import os, time, math, threading, argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table, callback, Input, Output, State, no_update

VAR_X = "py_x"  
VAR_Y = "py_y"
VAR_Z = "py_z"

N_SAMPLES_DEFAULT = 1000
SNAP_DIR = Path("snapshots"); SNAP_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("chunks");    DATA_DIR.mkdir(exist_ok=True)

ingress = deque(maxlen=200_000)           
to_chunk = deque()                       

latest = {"x": None, "y": None, "z": None}
_last_written = (None, None, None)
chunk_ready = threading.Event()
last_chunk_path = ""                      
stop_event = threading.Event()

def _write_snapshot_csv(df: pd.DataFrame) -> str:
    """Atomic CSV write to avoid partial reads on Windows."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = DATA_DIR / f"accel_chunk_{ts}.csv"
    tmp   = DATA_DIR / f".accel_chunk_{ts}.csv.part"
    df.to_csv(tmp, index=False)
    tmp.replace(final)
    return str(final)

def _save_plot_image(fig, ts_label: str) -> str:
    """Save a snapshot image (PNG via kaleido if available, else HTML)."""
    png_path = SNAP_DIR / f"accel_{ts_label}.png"
    html_path = SNAP_DIR / f"accel_{ts_label}.html"
    try:
        fig.write_image(str(png_path), scale=2, width=1200, height=700)
        return str(png_path)
    except Exception:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        return str(html_path)

def _stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["x", "y", "z", "mag"]:
        s = df[col].astype(float)
        rows.append({
            "metric": col,
            "mean": float(s.mean()),
            "std":  float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min":  float(s.min()),
            "max":  float(s.max()),
            "ptp":  float(s.max() - s.min()),
        })
    return pd.DataFrame(rows)

def _fig_xyz(df: pd.DataFrame):
    f = go.Figure()
    f.add_trace(go.Scatter(x=df["timestamp"], y=df["x"], mode="lines", name="X"))
    f.add_trace(go.Scatter(x=df["timestamp"], y=df["y"], mode="lines", name="Y"))
    f.add_trace(go.Scatter(x=df["timestamp"], y=df["z"], mode="lines", name="Z"))
    f.update_layout(
        title="Accelerometer X/Y/Z",
        xaxis_title="Time",
        yaxis_title="Acceleration",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return f

def _fig_mag(df: pd.DataFrame):
    f = go.Figure()
    f.add_trace(go.Scatter(x=df["timestamp"], y=df["mag"], mode="lines", name="|a|"))
    f.update_layout(
        title="Acceleration Magnitude |a| = sqrt(x² + y² + z²)",
        xaxis_title="Time",
        yaxis_title="Magnitude",
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False
    )
    return f

def _append_sample():
    """Append a new (timestamp, x, y, z) to both queues when x,y,z are ready and changed."""
    global _last_written
    if None in latest.values():
        return
    triple = (latest["x"], latest["y"], latest["z"])
    if triple == _last_written:
        return
    _last_written = triple
    sample = (datetime.now().astimezone(), *triple)
    ingress.append(sample)    
    to_chunk.append(sample)  
    if len(to_chunk) % 50 == 0:
        print(f"[Ingress] to_chunk queued: {len(to_chunk)}  | preview size: {len(ingress)}")

def cloud_thread(n_samples: int, test_mode: bool = False):
    """Fill queues with live Cloud data (py_x/py_y/py_z) or synthetic test data."""
    if test_mode:
        print("[TEST] Generating synthetic data…")
        t = 0.0
        dt = 0.02   
        phase = 0
        while not stop_event.is_set():
            if phase % 2 == 0:
                x = 0.8 * math.sin(2 * math.pi * 1.5 * t) + 0.05 * np.random.randn()
                y = 0.8 * math.cos(2 * math.pi * 1.5 * t) + 0.05 * np.random.randn()
                z = 9.81 + 0.1 * np.random.randn()
            else:
                x = 0.05 * np.random.randn()
                y = 0.05 * np.random.randn()
                z = 9.81 + 1.2 * math.sin(2 * math.pi * 0.3 * t) + 0.05 * np.random.randn()
            latest["x"], latest["y"], latest["z"] = x, y, z
            _append_sample()
            t += dt
            time.sleep(dt)
            if t >= 12.0: 
                phase += 1
                t = 0.0
        return

    try:
        from arduino_iot_cloud import ArduinoCloudClient
    except Exception:
        print("❌ arduino-iot-cloud not installed. Run: pip install arduino-iot-cloud")
        stop_event.set(); return

    DEVICE_ID = "65de9c82-3f77-4199-a7c9-185b5113885f"
    SECRET    = "ZfZP3ffL0VNtqndJN61s@Bluy"
    print("Using variables:", VAR_X, VAR_Y, VAR_Z)
    print("DEVICE_ID set:", bool(DEVICE_ID), "SECRET set:", bool(SECRET))
    if not DEVICE_ID or not SECRET:
        print("❌ Set ARDUINO_DEVICE_ID and ARDUINO_SECRET_KEY env vars in this terminal.")
        stop_event.set(); return

    def on_x(client, value):
        try: latest["x"] = float(value)
        except Exception: latest["x"] = value
        print("[Cloud] x:", latest["x"])
        _append_sample()

    def on_y(client, value):
        try: latest["y"] = float(value)
        except Exception: latest["y"] = value
        print("[Cloud] y:", latest["y"])
        _append_sample()

    def on_z(client, value):
        try: latest["z"] = float(value)
        except Exception: latest["z"] = value
        print("[Cloud] z:", latest["z"])
        _append_sample()

    client = ArduinoCloudClient(
        device_id=DEVICE_ID,
        username=DEVICE_ID,
        password=SECRET,
        sync_mode=True
    )
    client.register(VAR_X, value=None, on_write=on_x)
    client.register(VAR_Y, value=None, on_write=on_y)
    client.register(VAR_Z, value=None, on_write=on_z)

    try:
        client.start()
        print("✅ Connected to Arduino IoT Cloud. Streaming…")
    except Exception as e:
        print(f"❌ client.start() failed: {e}")
        stop_event.set(); return

    try:
        while not stop_event.is_set():
            client.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

def chunker_thread(n_samples: int):
    """Cut N-sized chunks from to_chunk into CSVs; signal Dash to refresh."""
    global last_chunk_path
    buf = []
    while not stop_event.is_set():
        while to_chunk:
            buf.append(to_chunk.popleft())
        if len(buf) >= n_samples:
            chunk = buf[:n_samples]
            buf = buf[n_samples:]
            df = pd.DataFrame(chunk, columns=["timestamp", "x", "y", "z"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
            last_chunk_path = _write_snapshot_csv(df)   
            print("[Chunk] Saved:", last_chunk_path)
            chunk_ready.set()
        else:
            time.sleep(0.2)

app = Dash(__name__)
app.title = "Arduino Cloud Accelerometer (Dash)"

app.layout = html.Div([
    html.H1("Smartphone Accelerometer — Plotly Dash"),
    html.Div(id="status-bar", style={"marginBottom": "0.5rem", "fontWeight": "600"}),

    dcc.Graph(id="xyz-graph"),
    dcc.Graph(id="mag-graph"),

    html.H3("Latest Statistics"),
    dash_table.DataTable(
        id="stats-table",
        columns=[{"name": c, "id": c} for c in ["metric", "mean", "std", "min", "max", "ptp"]],
        data=[],
        style_table={"overflowX": "auto"},
        style_cell={"padding": "6px", "textAlign": "center"},
        style_header={"fontWeight": "bold"}
    ),

    dcc.Interval(id="poll", interval=1200, n_intervals=0),
    dcc.Store(id="last-csv-path", data=""),
], style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "system-ui, Segoe UI, Arial"})

@callback(
    Output("status-bar", "children"),
    Output("xyz-graph", "figure"),
    Output("mag-graph", "figure"),
    Output("stats-table", "data"),
    Output("last-csv-path", "data"),
    Input("poll", "n_intervals"),
    State("last-csv-path", "data"),
    prevent_initial_call=False
)
def refresh(_n, last_path):
    def _read_csv_safe(path: str, tries=3, delay=0.05):
        for i in range(tries):
            try:
                return pd.read_csv(path, parse_dates=["timestamp"])
            except Exception as e:
                if i == tries - 1:
                    raise
                time.sleep(delay)

    if chunk_ready.is_set():
        chunk_ready.clear()
        new_path = str(last_chunk_path) if last_chunk_path else ""
        if new_path:
            try:
                df = _read_csv_safe(new_path)
                df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
                stats = _stats(df).round(3)
                ts_label = Path(new_path).stem.replace("accel_chunk_", "")
                img_path = _save_plot_image(_fig_xyz(df), ts_label)
                status = f"Loaded {Path(new_path).name} — Saved plot: {img_path}"
                return status, _fig_xyz(df), _fig_mag(df), stats.to_dict("records"), new_path
            except Exception as e:
                return f"Failed to load new chunk ({e}). Keeping previous view.", no_update, no_update, no_update, (last_path or "")

    if ingress:
        preview = list(ingress)[-500:]
        df = pd.DataFrame(preview, columns=["timestamp","x","y","z"])
        df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        status = f"Previewing {len(df)} live samples…"
        stats = _stats(df).round(3).to_dict("records")
        return status, _fig_xyz(df), _fig_mag(df), stats, (last_path or "")

    if last_path:
        try:
            df = _read_csv_safe(last_path)
            df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
            status = f"No new data — showing {Path(last_path).name} ({len(df)} samples)"
            return status, _fig_xyz(df), _fig_mag(df), _stats(df).round(3).to_dict("records"), last_path
        except Exception as e:
            return f"Couldn’t reload last chunk ({e}). Keeping previous view.", no_update, no_update, no_update, last_path

    return "Waiting for first samples…", no_update, no_update, no_update, (last_path or "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SAMPLES_DEFAULT, help="Samples per chunk")
    parser.add_argument("--test", action="store_true", help="Run without Cloud; simulate activities")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    t1 = threading.Thread(target=cloud_thread, args=(args.n, args.test), daemon=True)
    t2 = threading.Thread(target=chunker_thread, args=(args.n,), daemon=True)
    t1.start(); t2.start()

    print(f"Dash running on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        app.run(debug=False, host=args.host, port=args.port)
    finally:
        stop_event.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

if __name__ == "__main__":
    main()

