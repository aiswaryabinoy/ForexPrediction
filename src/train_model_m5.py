import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- CONFIG ----------
DATA_CSV = "../data/processed/eurusd_m5_prepared.csv"   # M5 prepared file
MODEL_PATH = "../models/model_rf_m5.pkl"                # NEW model file

os.makedirs("../models", exist_ok=True)

# ---------- 1. LOAD DATA ----------
df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])

feature_cols = [
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return_1", "ma_5", "ma_20",
]
target_col = "direction_next"

X = df[feature_cols]
y = df[target_col]

print("M5 Data shape:", X.shape, "Labels:", y.shape)

# ---------- 2. TRAIN/TEST SPLIT (NO SHUFFLE FOR TIME SERIES) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False   # keep time order
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# ---------- 3. TRAIN RANDOM FOREST (SAME HYPERPARAMS AS M1 MODEL) ----------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------- 4. EVALUATE ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nM5 Test accuracy:", acc)
print("\nClassification report (M5):")
print(classification_report(y_test, y_pred))

# ---------- 5. SAVE MODEL ----------
joblib.dump(model, MODEL_PATH)
print("\nM5 model saved to:", MODEL_PATH)
