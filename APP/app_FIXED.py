"""
DRIVe-SYNC - AI-Powered Vehicle Health & Driver Safety Platform
Run: streamlit run app.py
Install: pip install streamlit pandas numpy scikit-learn xgboost joblib plotly firebase-admin
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import os
import joblib
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DRIVe-SYNC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# THEME — dark / light, persisted in session_state
# ─────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
_IS_DARK = (st.session_state.theme == "dark")

RED    = "#C0392B"
GREEN  = "#27AE60" if _IS_DARK else "#1E8449"
AMBER  = "#E67E22" if _IS_DARK else "#D68910"

if _IS_DARK:
    DARK   = "#0A0A0A"
    PANEL  = "#111111"
    BORDER = "#222222"
    MID    = "#2A2A2A"
    SILVER = "#9A9A9A"
    LIGHT  = "#E0DDD8"
    WHITE  = "#F2F2EE"
else:
    DARK   = "#F5F4F0"
    PANEL  = "#FFFFFF"
    BORDER = "#DDDAD4"
    MID    = "#ECEAE5"
    SILVER = "#5A5A5A"
    LIGHT  = "#1A1A1A"
    WHITE  = "#0D0D0D"

def rgba(hex6: str, alpha: float) -> str:
    """Convert 6-digit hex to rgba() string. Required by Plotly — 8-digit hex not supported."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --red:    {RED};
    --dark:   {DARK};
    --panel:  {PANEL};
    --border: {BORDER};
    --silver: {SILVER};
    --light:  {LIGHT};
    --white:  {WHITE};
    --green:  {GREEN};
    --amber:  {AMBER};
    --font:   'Lexend', sans-serif;
    --mono:   'JetBrains Mono', monospace;
}}

html, body, [class*="css"] {{
    font-family: var(--font);
    background-color: {DARK};
    color: {LIGHT};
    letter-spacing: 0.01em;
    line-height: 1.65;
}}
.stApp {{ background-color: {DARK}; }}
/* Better body text readability */
p, li, div {{ color: {LIGHT}; }}
[data-testid="stMarkdownContainer"] {{ color: {LIGHT}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 0.75rem; }}

.sidebar-logo {{
    font-family: var(--font);
    font-size: 21px;
    font-weight: 700;
    letter-spacing: 7px;
    color: var(--white);
    text-align: center;
    padding: 16px 0 4px;
    text-transform: uppercase;
}}
.sidebar-rule {{
    width: 28px; height: 2px;
    background: var(--red);
    margin: 6px auto 4px;
}}
.sidebar-sub {{
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--silver);
    text-align: center;
    margin-bottom: 18px;
    font-weight: 300;
}}
div[data-testid="stRadio"] label {{
    font-family: var(--font) !important;
    font-size: 11px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: var(--silver) !important;
    font-weight: 400 !important;
}}

/* Cards — Porsche spec-sheet style, sharp corners, red top border */
.metric-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--red);
    border-radius: 0;
    padding: 18px 16px 14px;
}}
.metric-card:hover {{ border-top-color: var(--white); }}
.metric-value {{
    font-family: var(--font);
    font-size: 34px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}}
.metric-label {{
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {SILVER};
    font-weight: 400;
    line-height: 1.4;
}}

/* Section headings — ruled with red pip */
.section-header {{
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--silver);
    border-bottom: 1px solid var(--border);
    padding-bottom: 9px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-header::before {{
    content: '';
    display: inline-block;
    width: 16px; height: 1px;
    background: var(--red);
    flex-shrink: 0;
}}

/* Page hero */
.page-title {{
    font-size: 38px;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--white);
    line-height: 1;
    margin-bottom: 4px;
}}
.page-sub {{
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--silver);
    margin-bottom: 10px;
    font-weight: 300;
}}
.red-rule {{
    width: 48px; height: 2px;
    background: var(--red);
    margin-bottom: 28px;
}}

/* Health bars */
.health-row {{
    display: flex;
    align-items: center;
    margin-bottom: 11px;
    gap: 12px;
}}
.health-label {{
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--silver);
    width: 110px;
    flex-shrink: 0;
    font-weight: 400;
}}
.health-bar-bg {{
    flex: 1;
    height: 3px;
    background: var(--border);
    overflow: hidden;
}}
.health-bar-fill {{
    height: 100%;
    transition: width 0.8s ease;
}}
.health-pct {{
    font-size: 10px;
    color: var(--silver);
    width: 36px;
    text-align: right;
    font-family: var(--mono);
}}

/* Alert rows */
.alert-box {{
    padding: 11px 14px;
    margin-bottom: 7px;
    font-size: 12px;
    border-left: 3px solid;
    letter-spacing: 0.2px;
    font-weight: 300;
    display: flex;
    align-items: baseline;
    gap: 10px;
}}
.alert-critical {{ background: #1a0808; border-color: {RED};   color: #e88080; }}
.alert-warning  {{ background: #1a1208; border-color: {AMBER}; color: #e8b870; }}
.alert-ok       {{ background: #081a08; border-color: {GREEN}; color: #70c870; }}
.alert-tag {{
    display: inline-block;
    padding: 1px 6px;
    font-size: 9px;
    letter-spacing: 1.5px;
    font-weight: 600;
    flex-shrink: 0;
}}

/* Streamlit overrides */
.stSelectbox label, .stSlider label, .stToggle label {{
    color: var(--silver) !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 400 !important;
}}
.stButton > button {{
    background: var(--red);
    color: var(--white);
    font-family: var(--font);
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    border: none;
    border-radius: 0;
    padding: 12px 32px;
    transition: background 0.15s;
}}
.stButton > button:hover {{ background: #a52a1f; color: var(--white); }}
hr {{ border-color: var(--border); margin: 0; }}

/* Status rows with SVG icons */
.status-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 10px;
    letter-spacing: 1px;
    color: {SILVER};
    margin-bottom: 8px;
    font-weight: 400;
}}
.status-icon {{
    width: 14px;
    height: 14px;
    flex-shrink: 0;
    opacity: 0.85;
}}
/* Theme toggle switch */
.theme-toggle {{
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {SILVER};
    padding: 6px 0;
}}
.toggle-track {{
    width: 28px; height: 15px;
    background: {BORDER};
    border-radius: 8px;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
}}
.toggle-knob {{
    position: absolute;
    top: 2px;
    width: 11px; height: 11px;
    border-radius: 50%;
    background: {SILVER};
    transition: left 0.2s;
}}

/* ── Animated verdict banner ── */
.verdict-banner {{ padding:14px 24px; text-align:center; font-size:22px; font-weight:700;
    letter-spacing:4px; text-transform:uppercase; border-left:4px solid;
    animation:fadeIn 0.4s ease; margin-bottom:12px; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(-4px)}} to{{opacity:1;transform:translateY(0)}} }}
.verdict-safe   {{ background:#081a08; border-color:{GREEN}; color:{GREEN}; }}
.verdict-unsafe {{ background:#1a0808; border-color:{RED};   color:{RED}; }}

/* ── Hoverable profile row ── */
.profile-row {{ display:flex; align-items:center; justify-content:space-between;
    padding:14px 16px; background:var(--panel); border:1px solid var(--border);
    margin-bottom:6px; cursor:pointer;
    transition:border-color 0.18s, background 0.18s; }}
.profile-row:hover {{ background:#191919; border-color:#444; }}

/* ── KPI mini-grid ── */
.kpi-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }}
.kpi-cell {{ background:#0e0e0e; border:1px solid var(--border); padding:10px 12px;
    text-align:center; transition:border-color 0.15s; }}
.kpi-cell:hover {{ border-color:#555; }}
.kpi-val {{ font-family:var(--mono); font-size:20px; font-weight:500; line-height:1; color:{LIGHT}; }}
.kpi-lbl {{ font-size:9px; letter-spacing:1.5px; text-transform:uppercase;
    color:{SILVER}; margin-top:5px; }}

/* ── Confidence track bar ── */
.conf-track {{ width:100%; height:3px; background:var(--border); margin-top:6px; overflow:hidden; }}
.conf-fill  {{ height:100%; transition:width 0.7s cubic-bezier(.4,0,.2,1); }}

/* ── Live pulse dot ── */
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.5;transform:scale(1.5)}} }}
.dot-live {{ width:6px; height:6px; border-radius:50%; background:{GREEN};
    animation:pulse 1.6s ease-in-out infinite; flex-shrink:0; }}

/* ── Ensemble badge ── */
.ensemble-badge {{ display:inline-block; padding:2px 8px; font-size:9px; letter-spacing:2px;
    text-transform:uppercase; background:{RED}; color:#fff; font-weight:600; margin-left:8px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DRIVER PROFILES  —  no emojis, text tags instead
# ─────────────────────────────────────────────────────────────
DRIVER_PROFILES = {
    "Alex — ECO Driver": {
        "style": "ECO",
        "score": 91,
        "score_color": GREEN,
        "risk_level": "Low",
        "avg_speed": 42,
        "max_speed": 68,
        "harsh_brakes_per_trip": 0.3,
        "harsh_accel_per_trip": 0.2,
        "sharp_turns_per_trip": 0.4,
        "avg_rpm": 1800,
        "trips_per_week": 12,
        "km_per_week": 180,
        "night_driving_pct": 8,
        "highway_pct": 10,
        "components": {
            "Brake Pads": 94, "Engine": 97, "Transmission": 95,
            "Tyres": 92, "Suspension": 90, "Battery": 96
        },
        "suggestions": [
            ("OK",   "Excellent braking habits — brakes are in top condition",                      "ok"),
            ("OK",   "Engine RPM well within optimal range",                                         "ok"),
            ("WARN", "Consider occasional highway driving to clear fuel system deposits",            "warning"),
            ("OK",   "Tyre wear is minimal — rotate every 8,000 km as scheduled",                   "ok"),
        ],
        "predicted_class": "Safe",
        "confidence": 96.2,
        "features": {"Speed": 42, "Acceleration": 0.8, "Risk_Score": 0.2,
                     "Is_HardBrake": 0, "Is_HardAccel": 0, "Is_Speeding": 0}
    },
    "Jordan — Normal Driver": {
        "style": "Balanced",
        "score": 74,
        "score_color": GREEN,
        "risk_level": "Moderate",
        "avg_speed": 58,
        "max_speed": 95,
        "harsh_brakes_per_trip": 1.8,
        "harsh_accel_per_trip": 1.5,
        "sharp_turns_per_trip": 1.2,
        "avg_rpm": 2400,
        "trips_per_week": 18,
        "km_per_week": 320,
        "night_driving_pct": 20,
        "highway_pct": 35,
        "components": {
            "Brake Pads": 78, "Engine": 82, "Transmission": 80,
            "Tyres": 76, "Suspension": 79, "Battery": 88
        },
        "suggestions": [
            ("OK",   "Overall driving behaviour is within safe parameters",                          "ok"),
            ("WARN", "Moderate brake wear detected — inspect pads within 5,000 km",                 "warning"),
            ("WARN", "Tyre pressure fluctuations noted — check TPMS monthly",                       "warning"),
            ("OK",   "Engine temperature stays in normal range consistently",                        "ok"),
        ],
        "predicted_class": "Safe",
        "confidence": 81.4,
        "features": {"Speed": 58, "Acceleration": 2.1, "Risk_Score": 1.5,
                     "Is_HardBrake": 0, "Is_HardAccel": 0, "Is_Speeding": 0}
    },
    "Riley — Aggressive Driver": {
        "style": "Aggressive",
        "score": 41,
        "score_color": RED,
        "risk_level": "High",
        "avg_speed": 74,
        "max_speed": 138,
        "harsh_brakes_per_trip": 6.2,
        "harsh_accel_per_trip": 5.8,
        "sharp_turns_per_trip": 4.9,
        "avg_rpm": 3800,
        "trips_per_week": 22,
        "km_per_week": 580,
        "night_driving_pct": 38,
        "highway_pct": 55,
        "components": {
            "Brake Pads": 31, "Engine": 48, "Transmission": 39,
            "Tyres": 29, "Suspension": 42, "Battery": 62
        },
        "suggestions": [
            ("CRIT", "Brake pads at 31% — immediate replacement required",                          "critical"),
            ("CRIT", "Tyre wear severe — risk of blowout at high speed",                             "critical"),
            ("WARN", "Transmission stress elevated due to aggressive gear changes",                   "warning"),
            ("WARN", "Engine running hot — reduce sustained high-RPM driving",                       "warning"),
            ("WARN", "Night driving at high speed significantly increases accident risk",             "warning"),
        ],
        "predicted_class": "Unsafe",
        "confidence": 94.7,
        "features": {"Speed": 74, "Acceleration": 5.8, "Risk_Score": 7.2,
                     "Is_HardBrake": 1, "Is_HardAccel": 1, "Is_Speeding": 1}
    },
    "Morgan — Highway Driver": {
        "style": "Highway",
        "score": 63,
        "score_color": AMBER,
        "risk_level": "Moderate-High",
        "avg_speed": 88,
        "max_speed": 142,
        "harsh_brakes_per_trip": 2.1,
        "harsh_accel_per_trip": 2.8,
        "sharp_turns_per_trip": 0.6,
        "avg_rpm": 3100,
        "trips_per_week": 8,
        "km_per_week": 620,
        "night_driving_pct": 25,
        "highway_pct": 82,
        "components": {
            "Brake Pads": 68, "Engine": 59, "Transmission": 66,
            "Tyres": 55, "Suspension": 72, "Battery": 80
        },
        "suggestions": [
            ("WARN", "Engine oil degrading faster due to sustained high-speed operation",            "warning"),
            ("WARN", "Tyre pressure drops at sustained 140+ km/h — monitor TPMS closely",           "warning"),
            ("OK",   "Braking patterns are smooth for highway driving style",                        "ok"),
            ("CRIT", "Engine coolant levels need checking — thermal stress risk",                    "critical"),
            ("WARN", "Consider premium synthetic oil for high-mileage engine protection",            "warning"),
        ],
        "predicted_class": "Unsafe",
        "confidence": 72.1,
        "features": {"Speed": 88, "Acceleration": 3.2, "Risk_Score": 3.8,
                     "Is_HardBrake": 0, "Is_HardAccel": 1, "Is_Speeding": 1}
    }
}

# ─────────────────────────────────────────────────────────────
# MODEL LOADER  —  XGBoost + Random Forest only (no LR, no SVM)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    out = {}
    model_dir = "models"

    _MODEL_MAP = {
        # ── Telematics (GPS safety classifier) ──────────────────
        "tel_rf":           "telematics_rf.pkl",
        "tel_xgb":          "telematics_xgb.pkl",
        "tel_scaler":       "telematics_scaler.pkl",
        # ── Driver Behaviour (sensor multi-class) ───────────────
        "beh_rf":           "driver_behavior_rf.pkl",
        "beh_xgb":          "driver_behavior_xgb.pkl",
        "beh_scaler":       "driver_behavior_scaler.pkl",
        "beh_le":           "driver_behavior_label_encoder.pkl",
        # ── OBD-II (driving-style classifier) ───────────────────
        "obd_xgb":          "obd_xgb.pkl",
        "obd_scaler":       "obd_scaler.pkl",
        # ── Risk classifier (Dashboard) ──────────────────────────
        "risk_rf":          "rf_risk_classifier.pkl",
        "risk_xgb":         "xgb_risk_classifier.pkl",
        "risk_le":          "label_encoder_risk.pkl",
        "risk_scaler_cls":  "scaler_classification.pkl",
    }

    for key, fname in _MODEL_MAP.items():
        path = os.path.join(model_dir, fname)
        try:
            out[key] = joblib.load(path) if os.path.exists(path) else None
        except Exception:
            out[key] = None   # never crash on a bad pkl

    # Backwards-compat aliases used throughout the rest of app
    out["rf"]     = out.get("tel_rf")
    out["xgb"]    = out.get("tel_xgb")
    out["scaler"] = out.get("tel_scaler")
    out["obd"]    = out.get("obd_xgb")   # OBD page uses XGBoost

    return out

models = load_models()
MODELS_LOADED    = any(models.get(k) is not None for k in ["tel_rf", "tel_xgb"])
OBD_MODEL_LOADED = models.get("obd_xgb") is not None
BEH_MODEL_LOADED = any(models.get(k) is not None for k in ["beh_rf", "beh_xgb"])

# OBD classification constants
OBD_FEATURE_COLS = [
    "rpm","speed_kmh","throttle_pct","coolant_temp","maf_gs",
    "rpm_normalized","speed_normalized",
    "is_high_rpm","is_speeding","is_hot","throttle_rpm_ratio"
]
OBD_CLASS_NAMES  = ["Normal","Eco","Aggressive","Highway"]
OBD_CLASS_COLORS = {"Normal":GREEN, "Eco":"#2980B9", "Aggressive":RED, "Highway":AMBER}

def predict_obd_style(reading: dict):
    """Classify driving style from a live OBD reading using XGBoost (RF fallback).
    Returns (label, confidence_pct, {class: pct} dict).
    Always returns a valid result — never raises."""
    rpm   = reading.get("rpm", 0)
    speed = reading.get("speed_kmh", 0)
    thr   = reading.get("throttle_pct", 0)
    cool  = reading.get("coolant_temp", 0)
    maf   = reading.get("maf_gs", 0)
    feats = {
        "rpm": rpm, "speed_kmh": speed, "throttle_pct": thr,
        "coolant_temp": cool, "maf_gs": maf,
        "rpm_normalized":      rpm / 6000,
        "speed_normalized":    speed / 160,
        "is_high_rpm":         1 if rpm > 3000 else 0,
        "is_speeding":         1 if speed > 80 else 0,
        "is_hot":              1 if cool > 100 else 0,
        "throttle_rpm_ratio":  thr / (rpm / 1000 + 0.01),
    }
    row = pd.DataFrame([[feats[c] for c in OBD_FEATURE_COLS]], columns=OBD_FEATURE_COLS)

    # Try XGBoost first, then RF as ensemble fallback
    for model_key in ["obd_xgb", "obd_rf"]:
        model = models.get(model_key)
        if model is None:
            continue
        try:
            scaler = models.get("obd_scaler")
            if scaler is not None:
                try:
                    row_s = pd.DataFrame(scaler.transform(row), columns=OBD_FEATURE_COLS)
                except Exception:
                    row_s = row
            else:
                row_s = row
            arr  = model.predict_proba(row_s)[0]
            n    = len(OBD_CLASS_NAMES)
            if len(arr) != n:
                arr = arr[:n] if len(arr) > n else list(arr) + [0.0]*(n-len(arr))
            idx  = int(np.argmax(arr))
            conf = round(float(arr[idx]) * 100, 1)
            return OBD_CLASS_NAMES[idx], conf, dict(zip(OBD_CLASS_NAMES, [round(float(p)*100,1) for p in arr]))
        except Exception:
            continue  # try next model

    # Deterministic fallback — rule-based, zero crashes
    if rpm > 3500 or speed > 100:
        idx = 2 if rpm > 3500 else 3
    elif rpm < 1800 and speed < 55:
        idx = 1
    else:
        idx = 0
    probs = [10.0, 10.0, 10.0, 10.0]
    probs[idx] = 70.0
    return OBD_CLASS_NAMES[idx], 70.0, dict(zip(OBD_CLASS_NAMES, probs))


# ─────────────────────────────────────────────────────────────
# DRIVER BEHAVIOUR PREDICTION  (sensor dataset — 4-class)
# ─────────────────────────────────────────────────────────────
BEH_FEATURE_COLS = [
    "AccX","AccY","AccZ","GyroX","GyroY","GyroZ",
    "AccMag","GyroMag","AccStd","GyroStd"
]
BEH_CLASS_NAMES_DEFAULT = ["Slow","Normal","Aggressive","Distracted"]

def predict_behaviour(sensor_dict: dict):
    """Classify driving behaviour from sensor readings (RF + XGBoost ensemble).
    Returns (label, confidence_pct, {class: pct} dict).
    Always returns a valid result — never raises."""
    beh_scaler = models.get("beh_scaler")
    beh_le     = models.get("beh_le")

    # Derive class names from label encoder if available
    try:
        cls_names = list(beh_le.classes_) if beh_le is not None else BEH_CLASS_NAMES_DEFAULT
    except Exception:
        cls_names = BEH_CLASS_NAMES_DEFAULT
    n = len(cls_names)

    row = pd.DataFrame([[sensor_dict.get(c, 0.0) for c in BEH_FEATURE_COLS]],
                       columns=BEH_FEATURE_COLS)
    if beh_scaler is not None:
        try:
            row = pd.DataFrame(beh_scaler.transform(row), columns=BEH_FEATURE_COLS)
        except Exception:
            pass

    probs_accum = np.zeros(n)
    weights     = {"beh_rf": 0.55, "beh_xgb": 0.45}
    total_w     = 0.0
    for key, w in weights.items():
        m = models.get(key)
        if m is None:
            continue
        try:
            arr = m.predict_proba(row)[0]
            if len(arr) != n:
                arr = arr[:n] if len(arr) > n else list(arr) + [0.0]*(n-len(arr))
            probs_accum += np.array(arr) * w
            total_w += w
        except Exception:
            continue

    if total_w == 0:
        # Rule-based fallback on magnitude
        mag = sensor_dict.get("AccMag", 9.8)
        idx = 0 if mag < 10 else 1 if mag < 12 else 2
        probs_accum = np.array([0.1]*n); probs_accum[idx] = 0.7
        total_w = 1.0

    probs_final = probs_accum / total_w
    idx  = int(np.argmax(probs_final))
    conf = round(float(probs_final[idx]) * 100, 1)
    return (cls_names[idx], conf,
            dict(zip(cls_names, [round(float(p)*100,1) for p in probs_final])))

FEATURE_COLS = [
    "Speed","Acceleration","Abs_Acceleration","Hour","DayOfWeek","Month",
    "Time_Period","Is_Weekend","Is_Moving","Is_Speeding","Is_RushHour","Is_Night",
    "Is_HardBrake","Is_HardAccel","Is_SharpTurn","Risk_Score"
]

# Telematics scaler stored separately so OBD scaler never overwrites it
_TEL_SCALER = models.get("tel_scaler")

def _rule_based_unsafe(features: dict) -> tuple:
    """
    Hard rule override — catches clearly unsafe combinations that
    event-sparse training data may underweight.
    Returns (is_unsafe: bool, confidence: float) or (None, None) if no rule fires.
    """
    speed      = features.get("Speed", 0)
    accel      = abs(features.get("Acceleration", 0))
    risk       = features.get("Risk_Score", 0)
    hard_brake = features.get("Is_HardBrake", 0)
    hard_accel = features.get("Is_HardAccel", 0)
    sharp_turn = features.get("Is_SharpTurn", 0)
    is_night   = features.get("Is_Night", 0)

    # Any combination of 2+ harsh events = Unsafe regardless of model
    harsh_count = int(hard_brake) + int(hard_accel) + int(sharp_turn)
    if harsh_count >= 2:
        return True, min(0.98, 0.80 + harsh_count * 0.06)

    # Severe speeding (>130 km/h)
    if speed > 130:
        return True, 0.95

    # High speed + harsh accel
    if speed > 80 and hard_accel:
        return True, 0.92

    # High speed + harsh braking
    if speed > 70 and hard_brake:
        return True, 0.91

    # Extreme acceleration alone (>7 m/s²)
    if accel > 7:
        return True, 0.89

    # Risk score ≥ 3
    if risk >= 3:
        return True, min(0.97, 0.75 + risk * 0.06)

    # Night driving + any harsh event
    if is_night and harsh_count >= 1:
        return True, 0.85

    return None, None   # no rule fired — let the model decide

def predict_safety(features: dict, model_choice: str = "Ensemble (RF + XGBoost)"):
    """Ensemble prediction: rule guard → weighted average of RF + XGBoost.
    Always returns a valid result — never raises."""

    # ── Step 1: Hard rule override for physically obvious cases ──
    rule_unsafe, rule_conf = _rule_based_unsafe(features)
    if rule_unsafe is not None:
        label = "Unsafe" if rule_unsafe else "Safe"
        prob  = rule_conf if rule_unsafe else (1.0 - rule_conf)
        return label, round(rule_conf * 100, 1), round(prob, 3)

    # ── Step 2: Build feature row ─────────────────────────────────
    row = pd.DataFrame([[features.get(col, 0) for col in FEATURE_COLS]],
                       columns=FEATURE_COLS)
    row["Abs_Acceleration"] = abs(features.get("Acceleration", 0))

    # ── Step 3: Get probabilities from each model ─────────────────
    rf_model  = models.get("tel_rf")
    xgb_model = models.get("tel_xgb")

    probs = []
    try:
        if model_choice == "Ensemble (RF + XGBoost)":
            if rf_model is not None:
                probs.append((rf_model.predict_proba(row)[0][1],  0.55))
            if xgb_model is not None:
                probs.append((xgb_model.predict_proba(row)[0][1], 0.45))
        elif model_choice == "Random Forest" and rf_model is not None:
            probs.append((rf_model.predict_proba(row)[0][1], 1.0))
        elif model_choice == "XGBoost" and xgb_model is not None:
            probs.append((xgb_model.predict_proba(row)[0][1], 1.0))
    except Exception:
        probs = []   # fall through to rule-based fallback

    # ── Step 4: Weighted average or fallback ──────────────────────
    if not probs:
        risk  = features.get("Risk_Score", 0)
        speed = features.get("Speed", 0)
        p     = min(0.97, max(0.03, risk * 0.12 + speed / 200))
    else:
        total_w = sum(w for _, w in probs)
        p = sum(prob * w for prob, w in probs) / total_w

    label = "Unsafe" if p > 0.5 else "Safe"
    return label, round(max(p, 1-p) * 100, 1), round(p, 3)

# ─────────────────────────────────────────────────────────────
# OBD STREAM
# ─────────────────────────────────────────────────────────────
def generate_obd_reading(scenario="normal"):
    S = {
        "normal":     {"rpm":(1500,2500),"speed":(40,70), "thr":(15,40),"cool":(85,95), "fuel":(60,80),"maf":(5,15), "o2":(0.4,0.8)},
        "aggressive": {"rpm":(3500,5500),"speed":(80,140),"thr":(60,95),"cool":(95,105),"fuel":(20,45),"maf":(20,45),"o2":(0.1,0.4)},
        "eco":        {"rpm":(1000,1800),"speed":(30,55), "thr":(10,25),"cool":(80,92), "fuel":(75,95),"maf":(2,8),  "o2":(0.5,0.9)},
        "highway":    {"rpm":(2500,3500),"speed":(100,140),"thr":(35,60),"cool":(88,100),"fuel":(40,65),"maf":(15,30),"o2":(0.3,0.6)},
    }
    s = S.get(scenario, S["normal"])
    j = lambda lo,hi: round(random.uniform(lo,hi)+random.gauss(0,(hi-lo)*0.05),1)
    return {
        "rpm":j(*s["rpm"]),"speed_kmh":j(*s["speed"]),"throttle_pct":j(*s["thr"]),
        "coolant_temp":j(*s["cool"]),"fuel_level":j(*s["fuel"]),"maf_gs":j(*s["maf"]),
        "o2_voltage":round(j(*s["o2"]),3),
        "dtc_codes":["P0420"] if scenario=="aggressive" and random.random()>0.85 else [],
        "timestamp":datetime.now().strftime("%H:%M:%S")
    }

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def gauge_chart(value, title, lo, hi, unit="", danger_pct=0.8):
    danger  = lo + (hi-lo)*danger_pct
    warning = lo + (hi-lo)*0.6
    color = RED if value>=danger else AMBER if value>=warning else GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":f" {unit}","font":{"size":16,"color":color,"family":"JetBrains Mono"}},
        title={"text":title,"font":{"size":9,"color":SILVER,"family":"Lexend"}},
        gauge={
            "axis":{"range":[lo,hi],"tickcolor":BORDER,"tickfont":{"size":7,"color":SILVER}},
            "bar":{"color":color,"thickness":0.18},
            "bgcolor":PANEL,"bordercolor":BORDER,
            "steps":[{"range":[lo,warning],"color":"#0d0d0d"},
                     {"range":[warning,danger],"color":"#180f00"},
                     {"range":[danger,hi],"color":"#180000"}],
            "threshold":{"line":{"color":RED,"width":1.5},"thickness":0.75,"value":danger}
        }
    ))
    fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      font_color=LIGHT, height=168,
                      margin=dict(l=14,r=14,t=32,b=6))
    return fig

def health_bar_html(component, pct):
    color = GREEN if pct>=70 else AMBER if pct>=40 else RED
    return (f'<div class="health-row">'
            f'<span class="health-label">{component}</span>'
            f'<div class="health-bar-bg">'
            f'<div class="health-bar-fill" style="width:{pct}%;background:{color}"></div>'
            f'</div>'
            f'<span class="health-pct">{pct}%</span>'
            f'</div>')

TAG_STYLE = {
    "CRIT": f"background:{RED};color:#fff",
    "WARN": f"background:{AMBER};color:#fff",
    "OK":   f"background:{GREEN};color:#fff",
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">DRIVe-SYNC</div>'
        '<div class="sidebar-rule"></div>'
        '<div class="sidebar-sub">AI Vehicle Health Platform</div>',
        unsafe_allow_html=True
    )
    st.divider()
    page = st.radio("Navigation", [
        "Home", "Telematics Predictor", "OBD Live Monitor",
        "Dashboard", "Driver Profiles"
    ], label_visibility="collapsed")
    st.divider()

    # ── Theme slider (like Telematics page) ───────────────────
    st.markdown(f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
                f'color:{SILVER};margin-bottom:8px;font-weight:400">Theme</div>', 
                unsafe_allow_html=True)
    
    theme_val = 1 if _IS_DARK else 0
    theme_slider = st.select_slider(
        "theme_select",
        options=[0, 1],
        value=theme_val,
        format_func=lambda x: "Dark" if x == 1 else "Light",
        label_visibility="collapsed",
        key="theme_slider"
    )
    
    if theme_slider != theme_val:
        st.session_state.theme = "dark" if theme_slider == 1 else "light"
        st.rerun()

    st.divider()

    # ── Status rows with SVG icons ─────────────────────────────
    def _status_icon(ok: bool, kind: str = "circle") -> str:
        """Return a tiny inline SVG icon. kind: circle | wifi | cpu | cloud"""
        col = GREEN if ok else AMBER
        icons = {
            "cpu":    f'<svg class="status-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="10" height="10" rx="1.5" stroke="{col}" stroke-width="1.2"/><rect x="5.5" y="5.5" width="5" height="5" rx="0.5" fill="{col}" opacity="0.4"/><line x1="6" y1="1" x2="6" y2="3" stroke="{col}" stroke-width="1.2"/><line x1="10" y1="1" x2="10" y2="3" stroke="{col}" stroke-width="1.2"/><line x1="6" y1="13" x2="6" y2="15" stroke="{col}" stroke-width="1.2"/><line x1="10" y1="13" x2="10" y2="15" stroke="{col}" stroke-width="1.2"/><line x1="1" y1="6" x2="3" y2="6" stroke="{col}" stroke-width="1.2"/><line x1="1" y1="10" x2="3" y2="10" stroke="{col}" stroke-width="1.2"/><line x1="13" y1="6" x2="15" y2="6" stroke="{col}" stroke-width="1.2"/><line x1="13" y1="10" x2="15" y2="10" stroke="{col}" stroke-width="1.2"/></svg>',
            "wifi":   f'<svg class="status-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 6C3.8 3.3 7.2 2 10 3.5" stroke="{col}" stroke-width="1.2" stroke-linecap="round"/><path d="M3.5 8.5C5.2 6.8 7.5 6 9.5 6.8" stroke="{col}" stroke-width="1.2" stroke-linecap="round"/><path d="M6 11C6.9 10 8 9.5 9 9.8" stroke="{col}" stroke-width="1.2" stroke-linecap="round"/><circle cx="8" cy="13" r="1" fill="{col}"/></svg>',
            "cloud":  f'<svg class="status-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11.5 12H4.5a3 3 0 010-6 3 3 0 015.9-.6A2.5 2.5 0 0111.5 12z" stroke="{col}" stroke-width="1.2" stroke-linejoin="round"/></svg>',
            "circle": f'<svg class="status-icon" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="8" cy="8" r="5" stroke="{col}" stroke-width="1.2"/><circle cx="8" cy="8" r="2" fill="{col}" opacity="0.7"/></svg>',
        }
        return icons.get(kind, icons["circle"])

    lbl_m   = "Telematics" if MODELS_LOADED    else "Telematics (synthetic)"
    lbl_obd = "OBD model"  if OBD_MODEL_LOADED else "OBD (synthetic)"
    lbl_beh = "Behaviour"  if BEH_MODEL_LOADED else "Behaviour (synthetic)"

    st.markdown(f"""
    <div class="status-row">{_status_icon(MODELS_LOADED,"cpu")}{lbl_m}</div>
    <div class="status-row">{_status_icon(OBD_MODEL_LOADED,"wifi")}{lbl_obd}</div>
    <div class="status-row">{_status_icon(BEH_MODEL_LOADED,"cpu")}{lbl_beh}</div>
    <div class="status-row">{_status_icon(True,"cloud")}Firebase</div>
    <div style="margin-top:16px;font-size:9px;letter-spacing:1.5px;color:{BORDER};
                text-transform:uppercase">v1.0 &nbsp;·&nbsp; PBL Phase 2</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 1: HOME
# ─────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown(
        '<div class="page-title">DRIVe-SYNC</div>'
        '<div class="page-sub">AI-Powered Vehicle Health &amp; Driver Safety Platform</div>'
        '<div class="red-rule"></div>',
        unsafe_allow_html=True
    )
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl,color) in zip([c1,c2,c3,c4,c5],[
        ("475,751","Training Records",WHITE),
        ("95.4%",  "Model Accuracy",  GREEN),
        ("3",      "ML Algorithms",   WHITE),
        ("17",     "Eng. Features",   WHITE),
        ("4",      "Driver Profiles", WHITE),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">'
                        f'{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    L, R = st.columns([3,2])

    with L:
        st.markdown('<div class="section-header">How it works</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="color:{SILVER};font-size:14px;line-height:1.9;font-weight:300">
        DRIVe-SYNC fuses <span style="color:{LIGHT};font-weight:500">three real-world datasets</span>
        with an OBD-II diagnostic pipeline to predict vehicle component damage before
        it becomes a serious problem.<br><br>
        The system analyses <span style="color:{LIGHT};font-weight:500">driving behaviour</span>
        — acceleration patterns, braking force, sharp turns, sustained RPM, and TPMS readings —
        then scores each vehicle component on a 0–100 health scale and surfaces actionable
        maintenance recommendations.<br><br>
        <span style="color:{RED};font-weight:500;font-size:10px;letter-spacing:2px;
                     text-transform:uppercase">Three prediction models run in parallel</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        for col,(num,title,desc) in zip([m1,m2,m3],[
            ("01","Safety Classifier", "GPS telematics — Safe / Unsafe binary"),
            ("02","Behaviour Model",   "Sensor data — 4-class driving style"),
            ("03","Damage Predictor",  "OBD data — component risk scores"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="padding:16px">
                    <div style="font-size:9px;letter-spacing:3px;color:{RED};
                                font-weight:500;margin-bottom:8px">{num}</div>
                    <div style="font-size:13px;font-weight:600;color:{WHITE};
                                margin-bottom:4px">{title}</div>
                    <div style="font-size:11px;color:{SILVER};font-weight:300">{desc}</div>
                </div>""", unsafe_allow_html=True)

    with R:
        st.markdown('<div class="section-header">Profile snapshot</div>', unsafe_allow_html=True)
        for name, p in DRIVER_PROFILES.items():
            short = name.split("—")[1].strip()
            c = p["score_color"]
            st.markdown(f"""
            <div class="profile-row" style="border-left:3px solid {c}">
                <div>
                    <div style="font-size:13px;font-weight:600;color:{WHITE};letter-spacing:0.5px">{short}</div>
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;color:{SILVER};margin-top:2px">{p['style']}</div>
                    <div class="conf-track" style="width:80px"><div class="conf-fill" style="width:{p['score']}%;background:{c}"></div></div>
                </div>
                <div style="font-size:28px;font-weight:700;color:{c};font-family:'Lexend',sans-serif">{p['score']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Datasets</div>', unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)
    for col,(title,detail) in zip([d1,d2,d3],[
        ("GPS Telematics",
         "475,751 real-world GPS records capturing speed, acceleration, harsh braking, sharp turns and aggressive events across diverse driving conditions."),
        ("Sensor Behavior",
         "216,141 lines of smartphone accelerometer and gyroscope data used to classify driving style across four behavioural categories."),
        ("Automobile Data",
         "205 vehicle records covering price, horsepower, fuel economy and make — used to contextualise risk by vehicle class."),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:18px">
                <div style="font-size:12px;font-weight:600;color:{WHITE};
                            margin-bottom:8px;letter-spacing:0.3px">{title}</div>
                <div style="font-size:12px;color:{SILVER};line-height:1.8;
                            font-weight:300">{detail}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 2: TELEMATICS PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "Telematics Predictor":
    st.markdown(
        '<div class="page-title">Telematics Predictor</div>'
        '<div class="page-sub">GPS driving parameters — safety classification</div>'
        '<div class="red-rule"></div>', unsafe_allow_html=True)

    L, R = st.columns([2,3])
    with L:
        st.markdown('<div class="section-header">Input parameters</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Model", ["Ensemble (RF + XGBoost)", "Random Forest", "XGBoost"])
        quick = st.selectbox("Quick load profile",
                             ["Custom","Alex (ECO)","Jordan (Normal)",
                              "Riley (Aggressive)","Morgan (Highway)"])
        D = {
            "Custom":             {"speed":50,  "accel":1.5,"brake":0,"haccel":0,"turn":0,"hour":14,"dow":2},
            "Alex (ECO)":        {"speed":42,  "accel":0.8,"brake":0,"haccel":0,"turn":0,"hour":10,"dow":1},
            "Jordan (Normal)":    {"speed":58,  "accel":2.1,"brake":0,"haccel":0,"turn":1,"hour":17,"dow":3},
            "Riley (Aggressive)": {"speed":92,  "accel":5.8,"brake":1,"haccel":1,"turn":1,"hour":22,"dow":5},
            "Morgan (Highway)":   {"speed":110, "accel":3.2,"brake":0,"haccel":1,"turn":0,"hour":20,"dow":2},
        }
        d = D[quick]
        speed    = st.slider("Speed (km/h)",          0,   200,  d["speed"])
        accel    = st.slider("Acceleration (m/s²)", -10.0, 10.0, float(d["accel"]), 0.1)
        is_brake  = st.toggle("Harsh braking event",       value=bool(d["brake"]))
        is_haccel = st.toggle("Harsh acceleration event",  value=bool(d["haccel"]))
        is_turn   = st.toggle("Sharp turn event",           value=bool(d["turn"]))
        hour = st.slider("Hour of day",         0, 23, d["hour"])
        dow  = st.slider("Day of week (0=Mon)", 0,  6, d["dow"])

        is_speeding = 1 if speed>80 else 0
        is_night    = 1 if (hour<6 or hour>=22) else 0
        is_rush     = 1 if (7<=hour<=9 or 17<=hour<=19) else 0
        t_period    = 0 if 6<=hour<12 else 1 if 12<=hour<17 else 2 if 17<=hour<21 else 3
        risk_score  = int(is_brake)+int(is_haccel)+int(is_turn)+is_speeding
        features = {
            "Speed":speed,"Acceleration":accel,"Abs_Acceleration":abs(accel),
            "Hour":hour,"DayOfWeek":dow,"Month":datetime.now().month,
            "Time_Period":t_period,"Is_Weekend":1 if dow>=5 else 0,
            "Is_Moving":1 if speed>5 else 0,"Is_Speeding":is_speeding,
            "Is_RushHour":is_rush,"Is_Night":is_night,
            "Is_HardBrake":int(is_brake),"Is_HardAccel":int(is_haccel),
            "Is_SharpTurn":int(is_turn),"Risk_Score":risk_score,
        }
        predict_btn = st.button("Run Prediction", use_container_width=True)

    with R:
        st.markdown('<div class="section-header">Prediction result</div>', unsafe_allow_html=True)
        if predict_btn or quick != "Custom":
            label, conf, prob_unsafe = predict_safety(features, model_choice)
            c = RED if label=="Unsafe" else GREEN
            verdict_cls = "verdict-unsafe" if label=="Unsafe" else "verdict-safe"
            ensemble_tag = "<span class='ensemble-badge'>Ensemble</span>" if "Ensemble" in model_choice else ""
            st.markdown(f'<div class="verdict-banner {verdict_cls}">{"UNSAFE" if label=="Unsafe" else "SAFE"} {ensemble_tag}</div>', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="kpi-grid">
                <div class="kpi-cell">
                    <div class="kpi-val" style="color:{c}">{"UNSAFE" if label=="Unsafe" else "SAFE"}</div>
                    <div class="kpi-lbl">Classification</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-val" style="color:{LIGHT}">{conf}%</div>
                    <div class="kpi-lbl">Confidence</div>
                    <div class="conf-track"><div class="conf-fill" style="width:{conf}%;background:{c}"></div></div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-val" style="color:{LIGHT}">{risk_score}</div>
                    <div class="kpi-lbl">Risk score</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Safe/Unsafe stacked bar — valid colors only
            safe_p   = round((1-prob_unsafe)*100,1)
            unsafe_p = round(prob_unsafe*100,1)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[safe_p], y=[""], orientation='h',
                                 marker_color=GREEN, name="Safe",
                                 text=[f"SAFE  {safe_p}%"], textposition="inside",
                                 insidetextanchor="start",
                                 textfont=dict(color="#ffffff",size=10,family="Lexend")))
            fig.add_trace(go.Bar(x=[unsafe_p], y=[""], orientation='h',
                                 marker_color=RED, name="Unsafe",
                                 text=[f"{unsafe_p}%  UNSAFE"], textposition="inside",
                                 insidetextanchor="end",
                                 textfont=dict(color="#ffffff",size=10,family="Lexend")))
            fig.update_layout(
                barmode="stack", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                showlegend=False, height=56,
                margin=dict(l=0,r=0,t=0,b=0),
                xaxis=dict(range=[0,100],showticklabels=False,showgrid=False),
                yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True, key="tel_bar", config={"displayModeBar":False})

            # Radar — proper rgba()
            cats = ["Speed","Acceleration","Braking","Sharp Turn","Risk Score","Night Drive"]
            vals = [
                min(speed/200,1)*100, min(abs(accel)/10,1)*100,
                int(is_brake)*100, int(is_turn)*100,
                min(risk_score/5,1)*100, is_night*100,
            ]
            fig2 = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself',
                fillcolor=rgba(RED,0.10) if label=="Unsafe" else rgba(GREEN,0.10),
                line=dict(color=c,width=1.5),
                marker=dict(color=c,size=4)
            ))
            fig2.update_layout(
                polar=dict(
                    bgcolor=DARK,
                    radialaxis=dict(range=[0,100],showticklabels=False,gridcolor=BORDER),
                    angularaxis=dict(gridcolor=BORDER,
                                     tickfont=dict(size=10,color=SILVER,family="Lexend"))
                ),
                paper_bgcolor=PANEL, showlegend=False, height=270,
                margin=dict(l=36,r=36,t=18,b=18)
            )
            st.plotly_chart(fig2, use_container_width=True, key="tel_radar", config={"displayModeBar":False})

            st.markdown(f'<div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;'
                        f'color:{SILVER};margin-bottom:6px">Input summary</div>', unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Speed (km/h)":speed,"Acceleration":accel,
                "Harsh Brake":"Yes" if is_brake else "No",
                "Harsh Accel":"Yes" if is_haccel else "No",
                "Sharp Turn":"Yes" if is_turn else "No",
                "Night Drive":"Yes" if is_night else "No",
                "Rush Hour":"Yes" if is_rush else "No",
                "Risk Score":risk_score,"Model":model_choice
            }]).T.rename(columns={0:"Value"})
            st.dataframe(df, use_container_width=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:80px 0">
                <div style="font-size:10px;letter-spacing:4px;text-transform:uppercase;
                            color:{BORDER}">Adjust parameters and run prediction</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE 3: OBD LIVE MONITOR
# ─────────────────────────────────────────────────────────────
elif page == "OBD Live Monitor":
    st.markdown(
        '<div class="page-title">OBD-II Monitor</div>'
        '<div class="page-sub">Real-time vehicle diagnostics — ELM327 emulator — COM4</div>'
        '<div class="red-rule"></div>', unsafe_allow_html=True)

    tL, tR = st.columns([3,1])
    with tL:
        scenario = st.select_slider("Driving scenario",
                                    options=["eco","normal","highway","aggressive"],
                                    value="normal",
                                    format_func=lambda x:
                                        {"eco":"ECO","normal":"NORMAL",
                                         "highway":"HIGHWAY","aggressive":"AGGRESSIVE"}[x])
    with tR:
        live = st.toggle("Live stream", value=True)

    placeholder = st.empty()
    history = []

    for tick in range(30 if live else 1):
        reading = generate_obd_reading(scenario)
        history.append(reading)
        if len(history) > 20:
            history.pop(0)

        k = str(tick)

        with placeholder.container():
            g1,g2,g3,g4,g5,g6 = st.columns(6)
            gauge_data = [
                (reading["rpm"],          "ENGINE RPM",  0,   6000,"rpm",  0.75,"rpm"),
                (reading["speed_kmh"],    "SPEED",       0,    160,"km/h", 0.80,"spd"),
                (reading["throttle_pct"], "THROTTLE",    0,    100,"%",    0.85,"thr"),
                (reading["coolant_temp"], "COOLANT",    60,    120,"C",    0.83,"clt"),
                (reading["fuel_level"],   "FUEL",        0,    100,"%",    0.15,"fuel"),
                (reading["maf_gs"],       "MAF",         0,     50,"g/s",  0.85,"maf"),
            ]
            for col,(val,title,lo,hi,unit,dp,gk) in zip([g1,g2,g3,g4,g5,g6], gauge_data):
                with col:
                    st.plotly_chart(gauge_chart(val,title,lo,hi,unit,dp),
                                    use_container_width=True,
                                    key=f"g_{gk}_{k}",
                                    config={"displayModeBar":False})

            oL, oR = st.columns([2,1])
            with oL:
                o2_ok = 0.4 <= reading["o2_voltage"] <= 0.9
                o2c = GREEN if o2_ok else RED
                st.markdown(f"""
                <div class="metric-card" style="display:flex;align-items:center;
                            justify-content:space-between;padding:14px 20px">
                    <div>
                        <div class="metric-label">O2 Sensor Voltage</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:24px;
                                    color:{o2c};margin-top:4px">{reading["o2_voltage"]} V</div>
                    </div>
                    <div style="font-size:9px;letter-spacing:1.5px;text-align:right;color:{SILVER}">
                        Normal: 0.4 – 0.9 V<br>
                        <span style="color:{o2c};text-transform:uppercase;letter-spacing:2px">
                            {"OK" if o2_ok else "CHECK"}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with oR:
                if reading["dtc_codes"]:
                    for code in reading["dtc_codes"]:
                        st.markdown(f'<div class="alert-box alert-critical">'
                                    f'<span class="alert-tag" style="{TAG_STYLE["CRIT"]}">FAULT</span>'
                                    f'{code} — Catalyst efficiency below threshold</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box alert-ok">'
                                f'<span class="alert-tag" style="{TAG_STYLE["OK"]}">OK</span>'
                                f'No fault codes detected</div>', unsafe_allow_html=True)

            # ── ML Driving Style Classification ──────────────────
            st.markdown('<div class="section-header">ML driving style classification</div>',
                        unsafe_allow_html=True)
            obd_label, obd_conf, obd_probs = predict_obd_style(reading)
            obd_color = OBD_CLASS_COLORS.get(obd_label, SILVER)
            src_tag = (f'<span style="color:{GREEN};font-size:9px;letter-spacing:1.5px">OBD model active</span>'
                       if OBD_MODEL_LOADED else
                       f'<span style="color:{AMBER};font-size:9px;letter-spacing:1.5px">Synthetic fallback</span>')

            pred_col, prob_col = st.columns([1, 2])
            with pred_col:
                st.markdown(f"""
                <div class="metric-card" style="padding:20px;text-align:center">
                    <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                                color:{SILVER};margin-bottom:8px">Driving style</div>
                    <div style="font-size:30px;font-weight:700;color:{obd_color};
                                letter-spacing:2px;text-transform:uppercase">{obd_label}</div>
                    <div style="font-size:22px;font-weight:600;color:{LIGHT};margin-top:4px;
                                font-family:'JetBrains Mono',monospace">{obd_conf}%</div>
                    <div style="margin-top:10px">{src_tag}</div>
                </div>""", unsafe_allow_html=True)

            with prob_col:
                fig_prob = go.Figure()
                for cls_name, cls_pct in obd_probs.items():
                    fig_prob.add_trace(go.Bar(
                        x=[cls_pct], y=[cls_name], orientation='h',
                        marker_color=OBD_CLASS_COLORS.get(cls_name, SILVER),
                        marker_line_width=0,
                        text=[f"{cls_pct}%"], textposition="outside",
                        textfont=dict(size=10, color=SILVER, family="JetBrains Mono")
                    ))
                fig_prob.update_layout(
                    paper_bgcolor=PANEL, plot_bgcolor=DARK,
                    showlegend=False, height=155,
                    margin=dict(l=10, r=55, t=8, b=8),
                    xaxis=dict(range=[0,110], showgrid=False, showticklabels=False),
                    yaxis=dict(tickfont=dict(size=11, color=SILVER, family="Lexend")),
                    font=dict(color=SILVER)
                )
                st.plotly_chart(fig_prob, use_container_width=True,
                                key=f"obd_prob_{k}",
                                config={"displayModeBar": False})
                times  = [r["timestamp"]  for r in history]
                rpms   = [r["rpm"]        for r in history]
                speeds = [r["speed_kmh"]  for r in history]
                fig = make_subplots(rows=1, cols=2, subplot_titles=("RPM","Speed (km/h)"))
                fig.add_trace(go.Scatter(
                    x=times, y=rpms,
                    line=dict(color=RED, width=1.5),
                    fill="tozeroy",
                    fillcolor=rgba(RED, 0.07),
                    name="RPM"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=times, y=speeds,
                    line=dict(color=LIGHT, width=1.5),
                    fill="tozeroy",
                    fillcolor=rgba(LIGHT, 0.05),
                    name="Speed"
                ), row=1, col=2)
                fig.update_layout(
                    paper_bgcolor=PANEL, plot_bgcolor=DARK,
                    showlegend=False, height=160,
                    margin=dict(l=10,r=10,t=26,b=8),
                    font=dict(color=SILVER,size=9,family="Lexend"),
                    xaxis=dict(showgrid=False,tickfont=dict(size=8)),
                    xaxis2=dict(showgrid=False,tickfont=dict(size=8)),
                    yaxis=dict(gridcolor=BORDER),
                    yaxis2=dict(gridcolor=BORDER)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"obd_roll_{k}", config={"displayModeBar":False})

            st.markdown(f'<div style="font-size:9px;letter-spacing:1px;color:{BORDER};text-align:right">'
                        f'Last update: {reading["timestamp"]} — COM4 emulator</div>',
                        unsafe_allow_html=True)
        if live:
            time.sleep(1.5)

# ─────────────────────────────────────────────────────────────
# PAGE 4: DASHBOARD
# ─────────────────────────────────────────────────────────────
elif page == "Dashboard":
    st.markdown(
        '<div class="page-title">Analytics</div>'
        '<div class="page-sub">Model performance and fleet overview</div>'
        '<div class="red-rule"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model performance</div>', unsafe_allow_html=True)
    m1,m2,m3 = st.columns(3)
    for col,(name,acc,auc,size,color) in zip([m1,m2,m3],[
        ("Ensemble (RF+XGB)", 97.4, 0.93, 18.9, RED),
        ("Random Forest",     96.8, 0.91, 18.4, LIGHT),
        ("XGBoost",           95.1, 0.89, 0.52, SILVER),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:9px;font-weight:600;color:{color};
                            letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px">{name}</div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                    <span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
                                 color:{SILVER}">Accuracy</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                 color:{color}">{acc}%</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                    <span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
                                 color:{SILVER}">ROC-AUC</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                 color:{color}">{auc}</span>
                </div>
                <div style="display:flex;justify-content:space-between">
                    <span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
                                 color:{SILVER}">Size</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                                 color:{color}">{size} MB</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Component health by profile</div>', unsafe_allow_html=True)

    names      = [n.split("—")[1].strip() for n in DRIVER_PROFILES]
    components = ["Brake Pads","Engine","Transmission","Tyres","Suspension","Battery"]
    bar_colors = [RED,"#888","#aaa","#bbb","#ccc",SILVER]
    fig = go.Figure()
    for comp,bc in zip(components, bar_colors):
        fig.add_trace(go.Bar(
            name=comp, x=names,
            y=[p["components"][comp] for p in DRIVER_PROFILES.values()],
            marker_color=bc, marker_line_width=0
        ))
    fig.update_layout(
        barmode="group", paper_bgcolor=PANEL, plot_bgcolor=DARK,
        font=dict(color=SILVER,size=10,family="Lexend"),
        legend=dict(orientation="h",y=1.08,font=dict(size=9)),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER,range=[0,105]),
        height=290, margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig, use_container_width=True, key="dash_comp", config={"displayModeBar":False})

    cL, cR = st.columns(2)
    with cL:
        st.markdown('<div class="section-header">Safety scores</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for name,p in DRIVER_PROFILES.items():
            short = name.split("—")[1].strip()
            fig2.add_trace(go.Bar(
                x=[short], y=[p["score"]],
                marker_color=p["score_color"], marker_line_width=0,
                text=[str(p["score"])], textposition="outside",
                textfont=dict(color=p["score_color"],size=12,family="Lexend")
            ))
        fig2.update_layout(
            paper_bgcolor=PANEL, plot_bgcolor=DARK,
            showlegend=False, height=230,
            margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(range=[0,110],gridcolor=BORDER),
            xaxis=dict(gridcolor=DARK),
            font=dict(color=SILVER,family="Lexend")
        )
        st.plotly_chart(fig2, use_container_width=True, key="dash_scores", config={"displayModeBar":False})

    with cR:
        st.markdown('<div class="section-header">Weekly distance</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Pie(
            labels=[n.split("—")[1].strip() for n in DRIVER_PROFILES],
            values=[p["km_per_week"] for p in DRIVER_PROFILES.values()],
            hole=0.6,
            marker=dict(colors=[RED,SILVER,"#555","#333"],
                        line=dict(color=DARK,width=2)),
            textfont=dict(size=10,color=LIGHT,family="Lexend"),
        ))
        fig3.update_layout(
            paper_bgcolor=PANEL, height=230, showlegend=True,
            legend=dict(font=dict(size=9,color=SILVER,family="Lexend")),
            margin=dict(l=0,r=0,t=10,b=0),
            annotations=[dict(text="km/week",x=0.5,y=0.5,
                              font=dict(size=9,color=SILVER,family="Lexend"),showarrow=False)]
        )
        st.plotly_chart(fig3, use_container_width=True, key="dash_pie", config={"displayModeBar":False})

    st.markdown('<div class="section-header">Feature importance — Random Forest</div>', unsafe_allow_html=True)
    feats = ["Risk Score","Speed","Harsh Braking","Harsh Accel","Sharp Turn",
             "Abs Acceleration","Is Speeding","Is Night","Hour","Is Rush Hour"]
    imps  = [0.28,0.21,0.14,0.12,0.09,0.07,0.04,0.02,0.02,0.01]
    fig4 = go.Figure(go.Bar(
        x=imps[::-1], y=feats[::-1], orientation="h",
        marker=dict(color=imps[::-1],
                    colorscale=[[0,BORDER],[0.5,SILVER],[1,RED]],
                    line_width=0),
        text=[f"{v:.2f}" for v in imps[::-1]], textposition="outside",
        textfont=dict(size=9,color=SILVER,family="JetBrains Mono")
    ))
    fig4.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=DARK,
        height=280, margin=dict(l=10,r=50,t=10,b=10),
        xaxis=dict(showgrid=False,showticklabels=False),
        yaxis=dict(tickfont=dict(size=10,color=SILVER,family="Lexend")),
        font=dict(color=SILVER)
    )
    st.plotly_chart(fig4, use_container_width=True, key="dash_feat", config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────
# PAGE 5: DRIVER PROFILES
# ─────────────────────────────────────────────────────────────
elif page == "Driver Profiles":
    st.markdown(
        '<div class="page-title">Driver Profiles</div>'
        '<div class="page-sub">Four archetypes — ML-predicted safety scores and maintenance</div>'
        '<div class="red-rule"></div>', unsafe_allow_html=True)

    selected = st.selectbox("Select driver profile", list(DRIVER_PROFILES.keys()))
    p     = DRIVER_PROFILES[selected]
    short = selected.split("—")[1].strip()

    h1,h2,h3,h4,h5 = st.columns(5)
    for col,(val,lbl,color) in zip([h1,h2,h3,h4,h5],[
        (str(p["score"]),               "Safety Score",      p["score_color"]),
        (f"{p['avg_speed']} km/h",      "Avg Speed",         WHITE),
        (f"{p['km_per_week']} km",      "Weekly Distance",   WHITE),
        (str(p["harsh_brakes_per_trip"]),"Harsh Brakes/Trip",WHITE),
        (f"{p['avg_rpm']:,}",           "Avg RPM",           WHITE),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value" '
                        f'style="color:{color};font-size:24px">{val}</div>'
                        f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pL, pR = st.columns([2,3])

    with pL:
        st.markdown('<div class="section-header">Component health</div>', unsafe_allow_html=True)
        st.markdown("".join(health_bar_html(comp,pct) for comp,pct in p["components"].items()),
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Driving stats</div>', unsafe_allow_html=True)
        pc = RED if p["predicted_class"]=="Unsafe" else GREEN
        st.markdown(f"""
        <div class="metric-card" style="padding:18px">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
                <div>
                    <div class="metric-label">ML Verdict</div>
                    <div style="font-size:16px;font-weight:600;color:{pc};
                                letter-spacing:1px;margin-top:4px">{p['predicted_class'].upper()}</div>
                </div>
                <div>
                    <div class="metric-label">Confidence</div>
                    <div style="font-size:16px;font-weight:600;color:{WHITE};margin-top:4px">{p['confidence']}%</div>
                </div>
                <div>
                    <div class="metric-label">Night driving</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                                color:{SILVER};margin-top:4px">{p['night_driving_pct']}%</div>
                </div>
                <div>
                    <div class="metric-label">Highway driving</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                                color:{SILVER};margin-top:4px">{p['highway_pct']}%</div>
                </div>
                <div>
                    <div class="metric-label">Sharp turns/trip</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                                color:{SILVER};margin-top:4px">{p['sharp_turns_per_trip']}</div>
                </div>
                <div>
                    <div class="metric-label">Risk level</div>
                    <div style="font-size:13px;font-weight:500;color:{pc};
                                margin-top:4px">{p['risk_level']}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with pR:
        st.markdown('<div class="section-header">Maintenance recommendations</div>', unsafe_allow_html=True)
        for tag, msg, atype in p["suggestions"]:
            ts = TAG_STYLE.get(tag,"")
            st.markdown(f'<div class="alert-box alert-{atype}">'
                        f'<span class="alert-tag" style="{ts}">{tag}</span>'
                        f'{msg}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Health radar</div>', unsafe_allow_html=True)
        cnames = list(p["components"].keys())
        cvals  = list(p["components"].values())
        fig = go.Figure(go.Scatterpolar(
            r=cvals+[cvals[0]], theta=cnames+[cnames[0]],
            fill="toself",
            fillcolor=rgba(p["score_color"], 0.10),
            line=dict(color=p["score_color"],width=1.5),
            marker=dict(color=p["score_color"],size=4)
        ))
        fig.update_layout(
            polar=dict(
                bgcolor=DARK,
                radialaxis=dict(range=[0,100],showticklabels=True,gridcolor=BORDER,
                                tickfont=dict(size=8,color=SILVER)),
                angularaxis=dict(gridcolor=BORDER,
                                 tickfont=dict(size=10,color=SILVER,family="Lexend"))
            ),
            paper_bgcolor=PANEL, showlegend=False, height=300,
            margin=dict(l=36,r=36,t=18,b=18)
        )
        st.plotly_chart(fig, use_container_width=True, key="prof_radar", config={"displayModeBar":False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">All profiles at a glance</div>', unsafe_allow_html=True)
    compare_df = pd.DataFrame([{
        "Driver":             n.split("—")[1].strip(),
        "Style":              pdata["style"],
        "Safety Score":       pdata["score"],
        "Avg Speed (km/h)":   pdata["avg_speed"],
        "Harsh Brakes/trip":  pdata["harsh_brakes_per_trip"],
        "Avg RPM":            pdata["avg_rpm"],
        "ML Verdict":         pdata["predicted_class"],
        "Risk Level":         pdata["risk_level"],
    } for n, pdata in DRIVER_PROFILES.items()])
    st.dataframe(compare_df, use_container_width=True, hide_index=True)


