import sys
import os
import pandas as pd

# ---------- CONFIG ----------
DEFAULT_CSV = "../data/raw/eurusd_m5_raw.csv"
OUTPUT_CSV = "../data/processed/eurusd_m5_prepared.csv"

os.makedirs("../data/processed", exist_ok=True)

# ---------- 1. READ RAW M5 DATA ----------
csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

# Format per row:
# 2024-07-29 22:25    1.08220    1.08226    1.08218    1.08223    115
df = pd.read_csv(
    csv_path,
    sep=r"\s+|\t+",   # support tabs or multiple spaces
    header=None,
    names=["timestamp", "open", "high", "low", "close", "volume"],
    engine="python",
)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Raw M5 sample:")
print(df.head())

# ---------- 2. FEATURES (same logic as M1) ----------
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

df["return_1"] = df["close"].pct_change()
df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()

df = df.dropna().reset_index(drop=True)

# ---------- 3. LABEL: NEXT CANDLE DIRECTION ----------
df["close_next"] = df["close"].shift(-1)
df = df.dropna().reset_index(drop=True)

df["direction_next"] = (df["close_next"] > df["close"]).astype(int)

# ---------- 4. SAVE ----------
final_cols = [
    "timestamp",
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return_1", "ma_5", "ma_20",
    "direction_next",
]

df_final = df[final_cols]
df_final.to_csv(OUTPUT_CSV, index=False)

print("Prepared M5 data saved:", OUTPUT_CSV)
print("Prepared M5 sample:")
print(df_final.head())
print("Total M5 rows:", len(df_final))
