import sys
import os
import pandas as pd

# ---------- 1. CONFIG ----------
DEFAULT_CSV = "../data/raw/eurusd_m1_raw.csv"
OUTPUT_CSV = "../data/processed/eurusd_m1_prepared.csv"

os.makedirs("../data/processed", exist_ok=True)

# ---------- 2. READ RAW CSV ----------
csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

# columns = timestamp, open, high, low, close, dummy
df = pd.read_csv(
    csv_path,
    header=None,
    names=["timestamp", "open", "high", "low", "close", "dummy"],
    parse_dates=["timestamp"],     # 👈 Now timestamp already exists!
    dtype={"open": float, "high": float, "low": float, "close": float}
)

# Remove dummy column
df = df.drop(columns=["dummy"])

print("Raw (clean) sample:")
print(df.head())

# ---------- 3. BUILD FEATURES ----------
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

df["return_1"] = df["close"].pct_change()
df["ma_5"] = df["close"].rolling(window=5).mean()
df["ma_20"] = df["close"].rolling(window=20).mean()

df = df.dropna().reset_index(drop=True)

# ---------- 4. LABEL (Next Candle Direction) ----------
df["close_next"] = df["close"].shift(-1)
df = df.dropna().reset_index(drop=True)
df["direction_next"] = (df["close_next"] > df["close"]).astype(int)

# ---------- 5. SAVE ----------
final_cols = [
    "timestamp",
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return_1", "ma_5", "ma_20",
    "direction_next"
]

df_final = df[final_cols]
df_final.to_csv(OUTPUT_CSV, index=False)

print("Prepared data saved:", OUTPUT_CSV)
print("Prepared sample:")
print(df_final.head())
print("Total rows:", len(df_final))
