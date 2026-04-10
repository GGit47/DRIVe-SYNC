"""
DRIVe-SYNC — Live OBD Reader (TCP mode)
========================================
This module replaces generate_obd_reading() in app.py with REAL
data from the ELM327 emulator via TCP — no serial port, no baudrate issues.

HOW TO USE
----------
Step 1 — Start the emulator in TCP mode (one terminal, leave it running):
    cd C:\\Users\\gokul\\Desktop\\PBL\\ELM327-emulator
    python -m elm -s car -n 35000

Step 2 — In app.py, add this import at the top:
    from obd_live_reader import get_live_reading, OBD_CONNECTED

Step 3 — In the OBD Live Monitor page, replace:
    reading = generate_obd_reading(scenario)
  with:
    reading = get_live_reading() if OBD_CONNECTED else generate_obd_reading(scenario)

That's it. The website will show real emulator data.

EMULATOR SCENARIO COMMANDS (type in emulator terminal while running):
    scenario city      — urban stop/start driving
    scenario highway   — sustained high speed
    scenario sport     — high RPM, aggressive
    scenario car       — default balanced car
"""

import obd
import threading
import time
import random
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TCP_URL       = "socket://localhost:35000"   # emulator TCP address
RECONNECT_SEC = 5                             # retry interval if disconnected
POLL_INTERVAL = 0.8                           # seconds between readings

# ─────────────────────────────────────────────────────────────
# GLOBAL STATE — thread-safe latest reading
# ─────────────────────────────────────────────────────────────
_latest_reading = None
_connected      = False
_lock           = threading.Lock()

def _make_empty_reading():
    return {
        "rpm": 0, "speed_kmh": 0, "throttle_pct": 0,
        "coolant_temp": 0, "fuel_level": 50, "maf_gs": 0,
        "o2_voltage": 0.7, "dtc_codes": [],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "source": "disconnected"
    }

# ─────────────────────────────────────────────────────────────
# OBD COMMANDS to poll
# ─────────────────────────────────────────────────────────────
COMMANDS = {
    "rpm":          obd.commands.RPM,
    "speed_kmh":    obd.commands.SPEED,
    "throttle_pct": obd.commands.THROTTLE_POS,
    "coolant_temp": obd.commands.COOLANT_TEMP,
    "maf_gs":       obd.commands.MAF,
    "fuel_level":   obd.commands.FUEL_LEVEL,
    "o2_voltage":   obd.commands.O2_B1S1,
}

def _safe_val(response, default=0):
    """Extract numeric magnitude from OBD response, return default if None."""
    try:
        if response and not response.is_null() and response.value is not None:
            v = response.value
            return round(float(v.magnitude if hasattr(v, "magnitude") else v), 2)
    except Exception:
        pass
    return default

def _query_all(conn):
    """Poll all commands and return a reading dict."""
    reading = {}
    for key, cmd in COMMANDS.items():
        try:
            resp = conn.query(cmd)
            reading[key] = _safe_val(resp)
        except Exception:
            reading[key] = 0

    # DTC fault codes
    try:
        dtc_resp = conn.query(obd.commands.GET_DTC)
        if dtc_resp and not dtc_resp.is_null() and dtc_resp.value:
            reading["dtc_codes"] = [str(c[0]) for c in dtc_resp.value]
        else:
            reading["dtc_codes"] = []
    except Exception:
        reading["dtc_codes"] = []

    reading["timestamp"] = datetime.now().strftime("%H:%M:%S")
    reading["source"]    = "live"
    return reading

# ─────────────────────────────────────────────────────────────
# BACKGROUND THREAD — keeps connection alive, updates _latest_reading
# ─────────────────────────────────────────────────────────────
def _background_loop():
    global _latest_reading, _connected
    conn = None

    while True:
        # ── Try to connect ────────────────────────────────────
        if conn is None or not conn.is_connected():
            try:
                print(f"[OBD] Connecting to {TCP_URL} ...")
                conn = obd.OBD(
                    TCP_URL,
                    fast=False,
                    timeout=3,
                    delay_cmds=0.1,
                    check_voltage=False
                )
                if conn.is_connected():
                    print("[OBD] Connected via TCP")
                    with _lock:
                        _connected = True
                else:
                    conn = None
                    with _lock:
                        _connected = False
                    time.sleep(RECONNECT_SEC)
                    continue
            except Exception as e:
                print(f"[OBD] Connection error: {e}")
                with _lock:
                    _connected = False
                time.sleep(RECONNECT_SEC)
                continue

        # ── Poll data ─────────────────────────────────────────
        try:
            reading = _query_all(conn)
            with _lock:
                _latest_reading = reading
        except Exception as e:
            print(f"[OBD] Poll error: {e}")
            with _lock:
                _connected = False
            try:
                conn.close()
            except Exception:
                pass
            conn = None

        time.sleep(POLL_INTERVAL)

# ─────────────────────────────────────────────────────────────
# START BACKGROUND THREAD (daemon — dies when app exits)
# ─────────────────────────────────────────────────────────────
_thread = threading.Thread(target=_background_loop, daemon=True)
_thread.start()

# ─────────────────────────────────────────────────────────────
# PUBLIC API — import these into app.py
# ─────────────────────────────────────────────────────────────
@property
def OBD_CONNECTED() -> bool:
    """True when the background thread has a live connection."""
    with _lock:
        return _connected

def get_live_reading() -> dict:
    """
    Return the most recent OBD reading from the emulator.
    Falls back to a zero reading if not connected.
    Call this instead of generate_obd_reading() in app.py.
    """
    with _lock:
        if _latest_reading is not None:
            return dict(_latest_reading)
    return _make_empty_reading()

def is_connected() -> bool:
    """Convenience function for Streamlit sidebar status check."""
    with _lock:
        return _connected
