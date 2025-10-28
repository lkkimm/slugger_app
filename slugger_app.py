
import math
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Config (Streamlit secrets)
# ----------------------------
# In Streamlit Cloud, add in "Settings → Secrets":
# api_key = <YOUR_SLUGGER_API_KEY>
# slugger_base_url = https://y1vw9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI
BASE_URL = st.secrets.get("slugger_base_url", "https://y1vw9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI")
API_KEY = st.secrets.get("api_key")  # required in deployed app
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

# Physics constants
G_FTPS2 = 32.174
MPH_TO_FTPS = 1.46667

st.set_page_config(page_title="SLUGGER Spray & Optimizer", layout="wide")
st.title("⚾ SLUGGER Spray Chart + Simple Optimizer")

with st.expander("How to use"):
    st.markdown("""
    1) In *Settings → Secrets*, set `api_key` (and optionally `slugger_base_url`).  
    2) Load games, copy a `game_id`, then load pitches.  
    3) If `x/y` are present in payload, we plot directly; otherwise we compute from `ev_mph` + `theta_deg`.  
    4) Run the simple grid-search optimizer for LF/CF/RF.
    """)

# For local dev convenience: allow entering key if not set in secrets
if not API_KEY:
    st.warning("No `api_key` found in Streamlit secrets. Enter a key for local testing (will not be saved).")
    manual_key = st.text_input("Enter x-api-key (local only)", type="password")
    if manual_key:
        API_KEY = manual_key
        HEADERS = {"x-api-key": API_KEY}

# ----------------------------
# Helper functions
# ----------------------------
def get_json(path, params=None, method="GET", json_body=None):
    url = f"{BASE_URL}{path}"
    if not API_KEY:
        st.stop()
    try:
        if method == "GET":
            r = requests.get(url, headers=HEADERS, params=params or {})
        else:
            r = requests.post(url, headers=HEADERS, json=json_body or {})
        r.raise_for_status()
        js = r.json()
        if not js.get("success", True):
            st.error(js.get("message", "API returned success=false"))
        return js
    except requests.HTTPError as e:
        st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        return {}
    except Exception as e:
        st.error(f"Request failed: {e}")
        return {}

def compute_physics(df):
    if not {"ev_mph","theta_deg"}.issubset(df.columns):
        return df.assign(hang_s=np.nan, x_ft=np.nan, y_ft=np.nan, hmax_ft=np.nan)
    v = df["ev_mph"].astype(float) * MPH_TO_FTPS
    theta = np.radians(df["theta_deg"].astype(float))
    vx = v * np.cos(theta); vy = v * np.sin(theta)
    t = np.where(vy > 0, 2.0*vy/G_FTPS2, 0.0)
    x = vx * t
    hmax = np.where(vy > 0, (vy**2)/(2.0*G_FTPS2), 0.0)
    out = df.copy()
    out["hang_s"] = np.clip(t, 0, None)
    out["x_ft"] = np.clip(x, 0, None)
    out["y_ft"] = 0.0
    out["hmax_ft"] = np.clip(hmax, 0, None)
    return out

def distance(a,b):
    dx = a[0]-b[0]; dy = a[1]-b[1]
    return math.hypot(dx,dy)

def catch_flag(ball_xy, fielder_xy, speed_ftps, hang_s):
    if hang_s is None or (isinstance(hang_s, float) and math.isnan(hang_s)):
        return 0
    return 1 if distance(ball_xy, fielder_xy) <= speed_ftps*hang_s else 0

def grid_search(landings_xy, hang_s, speed_ftps=28.0, x_min=150, x_max=350, x_step=10, y_min=-100, y_max=100, y_step=10):
    xs = list(range(int(x_min), int(x_max)+1, int(x_step)))
    ys = list(range(int(y_min), int(y_max)+1, int(y_step)))
    def score(px,py):
        return sum(catch_flag((bx,by), (px,py), speed_ftps, t) for (bx,by), t in zip(landings_xy, hang_s))
    best = {"LF":((None,None),-1),"CF":((None,None),-1),"RF":((None,None),-1)}
    for name in ["LF","CF","RF"]:
        for px in xs:
            for py in ys:
                s = score(px,py)
                if s>best[name][1]:
                    best[name]=((px,py),s)
    trio = {k:v[0] for k,v in best.items()}
    total = 0
    for (bx,by), t in zip(landings_xy, hang_s):
        c = (catch_flag((bx,by), trio["LF"], speed_ftps, t) or
             catch_flag((bx,by), trio["CF"], speed_ftps, t) or
             catch_flag((bx,by), trio["RF"], speed_ftps, t))
        total += 1 if c else 0
    return trio, total

# ----------------------------
# UI: Load Games
# ----------------------------
st.subheader("1) Browse Games")
c1, c2, c3, c4 = st.columns(4)
with c1: team_name = st.text_input("Team name (optional)")
with c2: ballpark_name = st.text_input("Ballpark (optional)")
with c3: date = st.text_input("Date (YYYY-MM-DD, optional)")
with c4: limit = st.number_input("Limit", min_value=1, max_value=1000, value=50, step=1)

if st.button("Load Games"):
    js = get_json("/games", params={
        "team_name": team_name or None,
        "ballpark_name": ballpark_name or None,
        "date": date or None,
        "limit": int(limit),
        "order": "DESC"
    })
    st.session_state["games"] = js.get("data", [])

games = st.session_state.get("games", [])
if games:
    st.dataframe(pd.DataFrame(games))

# ----------------------------
# UI: Pull Pitches & Plot
# ----------------------------
st.subheader("2) Load Pitches for a Game")
game_id = st.text_input("game_id")
pitch_limit = st.number_input("Pitch limit", min_value=1, max_value=1000, value=200, step=1)

if st.button("Load Pitches"):
    if not game_id:
        st.warning("Enter a game_id from the games table above.")
    else:
        js = get_json("/pitches", params={"game_id": game_id, "limit": int(pitch_limit)})
        st.session_state["pitches"] = js.get("data", [])

pitches = st.session_state.get("pitches", [])
if pitches:
    df = pd.DataFrame(pitches)
    st.write("Raw Pitches (head):")
    st.dataframe(df.head(20))

    # Resolve coordinates
    x_col = next((c for c in df.columns if c.lower() in {"x","x_ft","landing_x","spray_x"}), None)
    y_col = next((c for c in df.columns if c.lower() in {"y","y_ft","landing_y","spray_y"}), None)

    if x_col and y_col:
        xs = df[x_col].astype(float).fillna(0).tolist()
        ys = df[y_col].astype(float).fillna(0).tolist()
        df["x_ft"] = xs; df["y_ft"] = ys
    else:
        # Compute from ev/theta if available
        ev_alias = next((c for c in df.columns if c.lower() in {"ev_mph","exit_velocity_mph","exit_velo_mph"}), None)
        la_alias = next((c for c in df.columns if c.lower() in {"theta_deg","launch_angle_deg","la_deg"}), None)
        if ev_alias and la_alias:
            tmp = df.rename(columns={ev_alias:"ev_mph", la_alias:"theta_deg"})
            df = compute_physics(tmp)
        else:
            st.info("No x/y fields and missing ev_mph + theta_deg; cannot compute spray.")
            st.stop()

    # Spray plot
    fig, ax = plt.subplots()
    ax.scatter(df["x_ft"], df.get("y_ft", pd.Series([0.0]*len(df))), alpha=0.6)
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")
    ax.set_title("Spray Chart")
    st.pyplot(fig)

    # ----------------------------
    # Optimization
    # ----------------------------
    st.subheader("3) Simple Outfield Optimization")
    speed = st.number_input("Fielder speed (ft/s)", min_value=10.0, max_value=40.0, value=28.0, step=0.5)
    x_min = st.number_input("Grid x_min", value=150)
    x_max = st.number_input("Grid x_max", value=350)
    x_step = st.number_input("Grid x_step", value=10)
    y_min = st.number_input("Grid y_min", value=-100)
    y_max = st.number_input("Grid y_max", value=100)
    y_step = st.number_input("Grid y_step", value=10)

    if st.button("Optimize LF/CF/RF"):
        xs = df["x_ft"].fillna(0).tolist()
        ys = df.get("y_ft", pd.Series([0.0]*len(df))).fillna(0).tolist()
        landings = list(zip(xs, ys))
        hang_s = df.get("hang_s", pd.Series([1.5]*len(df))).fillna(1.5).tolist()
        trio, total = grid_search(landings, hang_s, speed_ftps=float(speed),
                                  x_min=int(x_min), x_max=int(x_max), x_step=int(x_step),
                                  y_min=int(y_min), y_max=int(y_max), y_step=int(y_step))
        rate = total / max(1, len(landings))
        st.write(f"LF: {trio['LF']}  |  CF: {trio['CF']}  |  RF: {trio['RF']}")
        st.write(f"Estimated catch rate: **{rate:.1%}**")

        fig2, ax2 = plt.subplots()
        ax2.scatter(xs, ys, alpha=0.6)
        ax2.scatter([trio['LF'][0], trio['CF'][0], trio['RF'][0]],
                    [trio['LF'][1], trio['CF'][1], trio['RF'][1]],
                    marker="x", s=100)
        ax2.set_xlabel("x (ft)"); ax2.set_ylabel("y (ft)")
        ax2.set_title("Spray with Optimal LF/CF/RF")
        st.pyplot(fig2)
