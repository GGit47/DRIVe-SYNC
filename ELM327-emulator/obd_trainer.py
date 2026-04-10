"""
DRIVe-SYNC — OBD Model Trainer
Run after obd_collector.py has created obd_training_data.csv

    python obd_trainer.py

Outputs:
    models/obd_safety_model.pkl
    models/obd_scaler.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DATA_FILE  = "obd_training_data.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 55)
print("DRIVe-SYNC — OBD Model Training")
print("=" * 55)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df):,} samples")
print(df["scenario"].value_counts().to_string())

# ── Features ─────────────────────────────────────────────────
FEATURES = ["rpm","speed_kmh","throttle_pct","coolant_temp","maf_gs"]

# Engineer extra features
df["rpm_normalized"]   = df["rpm"] / 6000
df["speed_normalized"] = df["speed_kmh"] / 160
df["is_high_rpm"]      = (df["rpm"] > 3000).astype(int)
df["is_speeding"]      = (df["speed_kmh"] > 80).astype(int)
df["is_hot"]           = (df["coolant_temp"] > 100).astype(int)
df["throttle_rpm_ratio"] = df["throttle_pct"] / (df["rpm"] / 1000 + 0.01)

FEATURES_EXT = FEATURES + [
    "rpm_normalized","speed_normalized",
    "is_high_rpm","is_speeding","is_hot","throttle_rpm_ratio"
]

X = df[FEATURES_EXT].fillna(0)
y = df["label"]   # 0=normal, 1=eco, 2=aggressive, 3=highway

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, f"{MODEL_DIR}/obd_scaler.pkl")

# ── Train ─────────────────────────────────────────────────────
print("\nTraining models...")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
joblib.dump(rf, f"{MODEL_DIR}/obd_safety_model.pkl")
print(f"  Random Forest  acc={rf_acc:.4f}  [SAVED as obd_safety_model.pkl]")

xgb = XGBClassifier(n_estimators=100, random_state=42,
                     eval_metric="mlogloss", use_label_encoder=False)
xgb.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
joblib.dump(xgb, f"{MODEL_DIR}/obd_xgb.pkl")
print(f"  XGBoost        acc={xgb_acc:.4f}  [saved as obd_xgb.pkl]")

# ── Report ────────────────────────────────────────────────────
best_model = rf if rf_acc >= xgb_acc else xgb
best_name  = "Random Forest" if rf_acc >= xgb_acc else "XGBoost"
print(f"\nBest model: {best_name} ({max(rf_acc,xgb_acc):.4f} accuracy)")

label_names = ["Normal","Eco","Aggressive","Highway"]
print("\nClassification report:")
print(classification_report(y_test, best_model.predict(X_test_s if best_model!=rf else X_test),
                             target_names=label_names))

print("\nFiles saved:")
for f in os.listdir(MODEL_DIR):
    size = os.path.getsize(f"{MODEL_DIR}/{f}") / 1024
    print(f"  {f:<35} {size:.1f} KB")
print("\nDone. OBD model ready for app.py integration.")
print("=" * 55)
