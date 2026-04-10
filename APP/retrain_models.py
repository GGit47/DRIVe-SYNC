import os, pandas as pd, numpy as np, joblib, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

DATA_FILE = "C:/Users/gokul/Desktop/PBL/final_data.csv"
os.makedirs("models", exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Loaded {len(df):,} records")

df["Position Time"] = pd.to_datetime(df["Position Time"], errors="coerce")
df["Hour"]        = df["Position Time"].dt.hour.fillna(12).astype(int)
df["DayOfWeek"]   = df["Position Time"].dt.dayofweek.fillna(0).astype(int)
df["Month"]       = df["Position Time"].dt.month.fillna(1).astype(int)
df["Time_Period"] = df["Hour"].apply(lambda h: 0 if 6<=h<12 else 1 if 12<=h<17 else 2 if 17<=h<21 else 3)
df["Is_Weekend"]  = (df["DayOfWeek"]>=5).astype(int)
df["Is_RushHour"] = df["Hour"].apply(lambda h: 1 if (7<=h<=9 or 17<=h<=19) else 0)
df["Is_Night"]    = df["Hour"].apply(lambda h: 1 if (h<6 or h>=22) else 0)
df["Speed"]       = pd.to_numeric(df["Speed"], errors="coerce").fillna(0)
df["Is_Moving"]   = (df["Speed"]>5).astype(int)
df["Is_Speeding"] = (df["Speed"]>80).astype(int)
df["Acceleration"]     = pd.to_numeric(df["Acceleration"], errors="coerce").fillna(0) if "Acceleration" in df.columns else df["Speed"].diff().fillna(0)
df["Abs_Acceleration"] = df["Acceleration"].abs()

for src, dst in [("Harsh Braking","Is_HardBrake"),("Harsh Acceleration","Is_HardAccel"),("Sharp Turn","Is_SharpTurn")]:
    df[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0).clip(0,1).astype(int) if src in df.columns else 0

df["Risk_Score"] = df["Is_HardBrake"] + df["Is_HardAccel"] + df["Is_SharpTurn"] + df["Is_Speeding"]
df["Is_Unsafe"]  = (pd.to_numeric(df["Aggressive Event"], errors="coerce").fillna(0)>0).astype(int) if "Aggressive Event" in df.columns else (df["Risk_Score"]>0).astype(int)
print(f"Safe: {(df.Is_Unsafe==0).sum():,}  Unsafe: {(df.Is_Unsafe==1).sum():,}")

FEATURES = ["Speed","Acceleration","Abs_Acceleration","Hour","DayOfWeek","Month",
            "Time_Period","Is_Weekend","Is_Moving","Is_Speeding","Is_RushHour",
            "Is_Night","Is_HardBrake","Is_HardAccel","Is_SharpTurn","Risk_Score"]

X = df[FEATURES].fillna(0)
y = df["Is_Unsafe"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler()
joblib.dump(sc.fit(Xtr), "models/telematics_scaler.pkl")
Xtrs, Xtes = sc.transform(Xtr), sc.transform(Xte)

print("Training Logistic Regression...", end=" ", flush=True)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(Xtrs, ytr)
joblib.dump(lr, "models/telematics_lr.pkl")
print(f"acc={accuracy_score(yte, lr.predict(Xtes)):.4f}")

print("Training Random Forest (10-15 mins)...", end=" ", flush=True)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(Xtr, ytr)
joblib.dump(rf, "models/telematics_rf.pkl")
print(f"acc={accuracy_score(yte, rf.predict(Xte)):.4f}")

print("Training XGBoost...", end=" ", flush=True)
sp = (ytr==0).sum() / max((ytr==1).sum(), 1)
xgb = XGBClassifier(n_estimators=100, scale_pos_weight=sp, random_state=42, eval_metric="logloss", use_label_encoder=False)
xgb.fit(Xtr, ytr)
joblib.dump(xgb, "models/telematics_xgb.pkl")
print(f"acc={accuracy_score(yte, xgb.predict(Xte)):.4f}")

print("\nDone. Models saved:")
for f in sorted(os.listdir("models")):
    print(f"  {f}  {os.path.getsize('models/'+f)/1024:.1f} KB")