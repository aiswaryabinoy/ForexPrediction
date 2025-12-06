import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score

# ================== CONFIG ==================

SYMBOL = "EUR/USD"

TIMEFRAME_CONFIG = {
    "M1": {
        "label": "1 minute",
        "interval": "1min",
        "model_path": "models/model_rf.pkl",
        "data_path": "data/processed/eurusd_m1_prepared.csv",
    },
    "M5": {
        "label": "5 minutes",
        "interval": "5min",
        "model_path": "models/model_rf_m5.pkl",
        "data_path": "data/processed/eurusd_m5_prepared.csv",
    },
}

API_KEY_PATH = "api.txt"  # file with your Twelve Data API key

FEATURE_COLS = [
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return_1", "ma_5", "ma_20",
]

# ================== HELPERS ==================

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


@st.cache_resource
def load_api_key(path: str = API_KEY_PATH) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        return f.read().strip()


def fetch_live_data_twelvedata(api_key: str, interval: str, n_points: int = 100) -> pd.DataFrame:
    """
    Fetch recent candles from Twelve Data.
    Returns DataFrame with columns: timestamp, open, high, low, close
    """
    if not api_key:
        raise ValueError("Twelve Data API key is empty or api.txt not found.")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": n_points,
        "apikey": api_key,
        "format": "JSON",
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "values" not in data:
        raise ValueError(f"Unexpected response from Twelve Data: {data}")

    records = []
    for item in data["values"]:
        records.append({
            "timestamp": pd.to_datetime(item["datetime"]),
            "open": float(item["open"]),
            "high": float(item["high"]),
            "low": float(item["low"]),
            "close": float(item["close"]),
        })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the same features we used for training.
    """
    df = df_raw.copy().sort_values("timestamp").reset_index(drop=True)

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["return_1"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    df = df.dropna().reset_index(drop=True)
    return df


def predict_direction(model, X_row):
    proba = model.predict_proba(X_row)[0]
    pred_class = int(np.argmax(proba))
    label = "BUY (Up)" if pred_class == 1 else "SELL (Down)"
    confidence = float(proba[pred_class])
    return label, confidence, pred_class


def plot_candles(df: pd.DataFrame, title: str):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="green",
                increasing_fillcolor="green",
                decreasing_line_color="red",
                decreasing_fillcolor="red",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(t=40, l=10, r=10, b=10),
        xaxis=dict(
            type="date",
            tickformat="%H:%M",
            tickangle=-45,
            ticks="outside",
            showgrid=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def compute_model_metrics(model_path: str, data_path: str):
    """
    Load model + data and compute accuracy and F1 score.
    Computed once per model and cached.
    """
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None

    model = joblib.load(model_path)
    df = pd.read_csv(data_path, parse_dates=["timestamp"])

    if "direction_next" not in df.columns:
        return None, None

    X = df[FEATURE_COLS]
    y = df["direction_next"]

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    return acc, f1

# ================== UI LAYOUT ==================

st.set_page_config(page_title="Forex Predictor – M1 & M5", layout="wide")

# top bar style
st.markdown(
    """
    <style>
    .big-signal {
        font-size: 32px;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        display: inline-block;
    }
    .buy {
        background-color: #16a34a22;
        color: #16a34a;
        border: 2px solid #16a34a;
    }
    .sell {
        background-color: #dc262622;
        color: #dc2626;
        border: 2px solid #dc2626;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = load_api_key()

col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("### ⚙️ Settings")

    timeframe = st.radio(
        "Timeframe",
        ["M1", "M5"],
        format_func=lambda x: f"{x} ({TIMEFRAME_CONFIG[x]['label']})",
    )

    candles_to_show = st.slider(
        "Candles to display",
        min_value=30,
        max_value=150,
        value=80,
        step=10,
    )

    if not api_key:
        st.error("API key not found or empty in api.txt (expected at api.txt).")
    else:
        st.success("API key loaded from file.")

    # show model metrics for selected timeframe
    cfg_left = TIMEFRAME_CONFIG[timeframe]
    acc, f1 = compute_model_metrics(cfg_left["model_path"], cfg_left["data_path"])
    st.markdown("### 📊 Model Performance")
    if acc is None or f1 is None:
        st.write("Metrics not available (check model and data files).")
    else:
        st.write(f"Accuracy: {acc:.2%}")
        st.write(f"F1 Score: {f1:.3f}")

    st.write("Model is trained on historical data for this timeframe.")
    go_button = st.button("🔃 Fetch data & Predict")

with col_right:
    st.markdown(f"## {SYMBOL} – Live Prediction")

    if not api_key:
        st.info("Fix API key file, then press 'Fetch data & Predict'.")
    elif go_button:
        cfg = TIMEFRAME_CONFIG[timeframe]

        try:
            # 120 candles to allow indicators
            df_live = fetch_live_data_twelvedata(api_key, cfg["interval"], n_points=120)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        else:
            try:
                model = load_model(cfg["model_path"])
            except Exception as e:
                st.error(f"Error loading model for {timeframe}: {e}")
            else:
                # chart
                st.markdown("#### Recent candles")
                plot_candles(df_live.tail(candles_to_show), f"{SYMBOL} – {timeframe} candles")

                # features + prediction
                feat_df = build_features(df_live)
                if feat_df.empty:
                    st.error("Not enough data to compute features.")
                else:
                    # LAST FEATURE ROW + only model columns
                    last_feat = feat_df.iloc[[-1]]
                    X_live = last_feat[FEATURE_COLS]

                    label, confidence, pred_class = predict_direction(model, X_live)

                    # last completed raw candle info
                    last_price_row = df_live.iloc[-1]
                    last_price = last_price_row["close"]
                    last_time = last_price_row["timestamp"]

                    # estimate volatility from recent ranges
                    recent_ranges = (df_live["high"] - df_live["low"]).tail(20)
                    avg_range = recent_ranges.mean()
                    if pd.isna(avg_range) or avg_range <= 0:
                        avg_range = (df_live["high"] - df_live["low"]).mean()
                    if pd.isna(avg_range) or avg_range <= 0:
                        avg_range = 0.0005  # fallback ~5 pips

                    pip_factor = 10000.0
                    sl_dist = avg_range               # in price
                    tp_dist = 2.0 * avg_range         # 2:1 R:R

                    tp_pips = tp_dist * pip_factor
                    sl_pips = sl_dist * pip_factor

                    # TP/SL prices depending on direction
                    if "BUY" in label:
                        tp_price = last_price + tp_dist
                        sl_price = last_price - sl_dist
                    else:  # SELL
                        tp_price = last_price - tp_dist
                        sl_price = last_price + sl_dist

                    # heuristic expected high/low for next candle
                    expected_high = max(last_price, tp_price)
                    expected_low = min(last_price, sl_price)

                    # breakout probability (reuse direction confidence)
                    breakout_prob = confidence

                    # expected profit / loss (in pips, based on TP/SL and confidence)
                    expected_profit_pips = confidence * tp_pips
                    expected_loss_pips = (1.0 - confidence) * sl_pips

                    risk_to_reward = tp_pips / sl_pips if sl_pips > 0 else None

                    # prediction panel
                    sig_class = "buy" if "BUY" in label else "sell"
                    st.markdown(
                        f"""
                        <div class="big-signal {sig_class}">
                            {label}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write(f"Confidence: {confidence:.2%}")
                    st.write(f"Last price: {last_price:.5f}")
                    st.write(f"Time (last candle): {last_time}")

                    st.markdown("#### Next Candle & Trade Metrics (Heuristic)")
                    st.write(f"Expected next candle high: {expected_high:.5f}")
                    st.write(f"Expected next candle low: {expected_low:.5f}")
                    st.write(f"Breakout probability (direction confidence): {breakout_prob:.2%}")

                    st.write(f"Suggested TP price: {tp_price:.5f} (~{tp_pips:.1f} pips)")
                    st.write(f"Suggested SL price: {sl_price:.5f} (~{sl_pips:.1f} pips)")

                    st.write(f"Expected profit (pips): {expected_profit_pips:.1f}")
                    st.write(f"Expected loss (pips): {expected_loss_pips:.1f}")
                    if risk_to_reward is not None:
                        st.write(f"Risk-to-Reward ratio: {risk_to_reward:.2f} : 1")

                    st.caption(
                        "Direction and confidence come from the trained ML model. "
                        "Expected high/low, TP/SL, and expected P/L are heuristic estimates "
                        "based on recent volatility and a 2:1 risk-to-reward assumption."
                    )
    else:
        st.write("Choose timeframe and press 'Fetch data & Predict'.")
