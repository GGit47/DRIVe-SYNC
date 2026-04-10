"""
DRIVe-SYNC — OBD Data Collector
Run AFTER starting ELM327 emulator on COM3:
    cd ELM327-emulator
    python elm.py -s car -p COM3

Then run this script:
    python obd_collector.py

Collects 4 driving scenarios and saves obd_training_data.csv
"""

import obd
import time
import csv
import random
from datetime import datetime

OUTPUT_FILE = "obd_training_data.csv"
PORT        = "COM4"          # Other end of VSPD pair
SAMPLES_PER_SCENARIO = 150    # ~2.5 min per scenario at 1s intervals

SCENARIOS = {
    "normal":     0,
    "eco":        1,
    "aggressive": 2,
    "highway":    3,
}

def collect_obd_data():
    print(f"\nDRIVe-SYNC OBD Collector")
    print(f"Connecting to {PORT}...")

    try:
        conn = obd.OBD(PORT, fast=False, timeout=5)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_obd(OUTPUT_FILE)

    if not conn.is_connected():
        print("OBD not connected — using synthetic fallback")
        return generate_synthetic_obd(OUTPUT_FILE)

    print(f"Connected. Collecting {SAMPLES_PER_SCENARIO} samples per scenario...\n")

    rows = []
    for scenario, label in SCENARIOS.items():
        print(f"  Scenario: {scenario.upper()} ({SAMPLES_PER_SCENARIO} samples)")
        for i in range(SAMPLES_PER_SCENARIO):
            try:
                rpm     = conn.query(obd.commands.RPM).value
                speed   = conn.query(obd.commands.SPEED).value
                throttle= conn.query(obd.commands.THROTTLE_POS).value
                coolant = conn.query(obd.commands.COOLANT_TEMP).value
                maf     = conn.query(obd.commands.MAF).value

                row = {
                    "timestamp":    datetime.now().isoformat(),
                    "scenario":     scenario,
                    "label":        label,
                    "rpm":          rpm.magnitude     if rpm     else 0,
                    "speed_kmh":    speed.magnitude   if speed   else 0,
                    "throttle_pct": throttle.magnitude if throttle else 0,
                    "coolant_temp": coolant.magnitude  if coolant  else 0,
                    "maf_gs":       maf.magnitude      if maf      else 0,
                }
                rows.append(row)
                if i % 30 == 0:
                    print(f"    [{i}/{SAMPLES_PER_SCENARIO}] RPM={row['rpm']:.0f} "
                          f"Speed={row['speed_kmh']:.0f} km/h")
            except Exception as e:
                print(f"    Sample error: {e}")
            time.sleep(1)

    _save_csv(rows, OUTPUT_FILE)
    print(f"\nSaved {len(rows)} samples to {OUTPUT_FILE}")

def generate_synthetic_obd(output_file):
    """Fallback: generate realistic synthetic OBD data for training."""
    print("Generating synthetic OBD training data...")
    profiles = {
        "normal":     {"rpm":(1500,2500),"speed":(40,70), "thr":(15,40),"cool":(85,95), "maf":(5,15)},
        "eco":        {"rpm":(1000,1800),"speed":(30,55), "thr":(10,25),"cool":(80,92), "maf":(2,8)},
        "aggressive": {"rpm":(3500,5500),"speed":(80,140),"thr":(60,95),"cool":(95,108),"maf":(20,45)},
        "highway":    {"rpm":(2500,3500),"speed":(100,140),"thr":(35,60),"cool":(88,100),"maf":(15,30)},
    }
    rows = []
    for scenario, label in SCENARIOS.items():
        p = profiles[scenario]
        for _ in range(SAMPLES_PER_SCENARIO):
            j = lambda lo,hi: round(random.uniform(lo,hi) + random.gauss(0,(hi-lo)*0.05), 2)
            rows.append({
                "timestamp":    datetime.now().isoformat(),
                "scenario":     scenario,
                "label":        label,
                "rpm":          j(*p["rpm"]),
                "speed_kmh":    j(*p["speed"]),
                "throttle_pct": j(*p["thr"]),
                "coolant_temp": j(*p["cool"]),
                "maf_gs":       j(*p["maf"]),
            })
    _save_csv(rows, output_file)
    print(f"Saved {len(rows)} synthetic samples to {output_file}")

def _save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    collect_obd_data()
