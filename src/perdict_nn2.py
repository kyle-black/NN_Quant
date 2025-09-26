#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from set_seed_all import seed_everything
from data_io import load_forex_csv
from features import add_basic_features
from pipelines import RollingScaler
from backtest import (
    softmax_to_dir_threshold,
    pnl_from_signals,
    trades_from_signals,
    summarize_trades,
    pnl_timeseries,            # daily PnL time series (all days)
)

# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
SEED = 42
seed_everything(SEED, deterministic_tf=True)

# =========================
# HARD-CODED PARAMETERS
# =========================
DATA_PATH   = "data/EURUSD_eod_2000_to_today.csv"   # daily file by default
MODELS_DIR  = "models/eurusd_nn"                    # has fold_*.keras + fold_*_feature_order.csv

# Match training window (Option A). Set to None to score full history.
RECENT_START = "2018-01-01"     # e.g., "2018-01-01" / "2020-01-01" / None

PMIN        = 0.00
MARGIN      = 0.05
COST_BP     = 0.0
HOLD_DAYS   = 21                                     # N bars if daily, 24*N if hourly
BARRIER_DATR_MULT = 1.0                             # None to disable; else k * daily ATR at entry
EXPORT_TRADES_CSV = "daily_export_trades.csv"
DAILY_PNL_CSV     = "daily_pnl_timeseries.csv"
# =========================

def _load_feature_cols(feat_path: str) -> list[str]:
    try:
        return pd.read_csv(feat_path)["feature"].astype(str).str.strip().tolist()
    except Exception:
        fc = pd.read_csv(feat_path, header=None).iloc[:,0].astype(str).str.strip().tolist()
        if fc in ([], ["0"], ["Unnamed: 0"], ["feature"]):
            raise RuntimeError(f"Feature list at {feat_path} looks invalid: {fc}.")
        return fc

def _infer_bars_per_day(index: pd.DatetimeIndex) -> int:
    freq = pd.infer_freq(index)
    if freq:
        f = freq.upper()
        if "D" in f: return 1
        if "H" in f: return 24
    if len(index) >= 3:
        dt = index.to_series().diff().median()
        if pd.Timedelta(hours=12) < dt < pd.Timedelta(days=2):  return 1
        if pd.Timedelta(minutes=30) < dt < pd.Timedelta(hours=2): return 24
    return 24

def _apply_recent_start(df: pd.DataFrame, recent_start: str | None) -> pd.DataFrame:
    if not recent_start:
        return df
    ts = pd.Timestamp(recent_start, tz=df.index.tz) if df.index.tz is not None else pd.Timestamp(recent_start)
    return df.loc[df.index >= ts].copy()

def predict_and_backtest(
    data_path: str,
    models_dir: str,
    pmin: float = 0.60,
    margin: float | None = 0.05,
    one_way_cost_bp: float = 0.5,
    export_trades_csv: str | None = None,
    hold_days: int | None = None,
    barrier_daily_atr_mult: float | None = None,
    recent_start: str | None = RECENT_START,   # <<< NEW: trim inference window
) -> dict:

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    keras_files = sorted(
        [f for f in os.listdir(models_dir) if f.endswith(".keras")],
        key=lambda n: int(n.split("_")[1].split(".")[0]) if "_" in n else 10**9
    )
    if not keras_files:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")

    # Load full data
    df_full = load_forex_csv(data_path)
    df_feat_full = add_basic_features(df_full)

    # Trim BOTH prices and features to recent window (keeps alignment)
    df = _apply_recent_start(df_full, recent_start)
    df_feat = _apply_recent_start(df_feat_full, recent_start)

    if df.empty or df_feat.empty:
        raise ValueError(f"No rows after applying RECENT_START={recent_start}. "
                         f"Check your date and input file.")

    print(f"[DATA]  {df.index.min()} → {df.index.max()}  rows={len(df)}  (recent_start={recent_start})")
    print(f"[FEAT]  {df_feat.index.min()} → {df_feat.index.max()}  rows={len(df_feat)})")

    # Proper OOS split size (e.g., 20% tail per fold)
    n = len(df_feat)
    test_size = 0.20
    test_n = int(max(1, n * test_size))
    n_folds = len(keras_files)

    # Convert hold_days -> bars (auto daily vs hourly) on the TRIMMED index
    if hold_days is not None:
        bars_per_day = _infer_bars_per_day(df_feat.index)
        hold_bars = int(hold_days * bars_per_day)
        print(f"[EXIT] HOLD_DAYS={hold_days} → bars_per_day={bars_per_day} → max_hold_bars={hold_bars}")
    else:
        hold_bars = None

    # Safety: only use barrier if daily ATR is present (in the trimmed features)
    if barrier_daily_atr_mult is not None and "atr_d1_14" not in df_feat.columns:
        print("[WARN] barrier_daily_atr_mult set but 'atr_d1_14' not in features; disabling barrier exits.")
        barrier_daily_atr_mult = None

    sigs = []
    for f in keras_files:
        fold = int(f.split("_")[1].split(".")[0]) if "_" in f else 999999999
        model = keras.models.load_model(os.path.join(models_dir, f))

        feat_path = os.path.join(models_dir, f"fold_{fold}_feature_order.csv")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature order file for {f}: {feat_path}")
        feature_cols = _load_feature_cols(feat_path)

        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            raise KeyError(f"Missing features in df_feat: {missing[:12]} ...")

        # Expanding-window style within the TRIMMED recent window
        end_train = int((n - test_n) * (fold / n_folds))
        end_train = max(end_train, 1)
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))
        if len(te) == 0 or len(tr) < 10:
            print(f"[FOLD {fold}] skipped (train too small or no test).")
            continue

        print(f"[FOLD {fold}] test slice: {df_feat.index[te][0]} → {df_feat.index[te][-1]}  (len={len(te)})")

        X = df_feat[feature_cols].copy()
        scaler = RollingScaler()
        scaler.fit(X.iloc[tr])
        X_te = scaler.transform(X.iloc[te])

        prob = model.predict(X_te, verbose=0)
        dir_sig = softmax_to_dir_threshold(prob, pmin=pmin, margin=margin)
        sigs.append(pd.Series(dir_sig, index=df_feat.index[te]))

    if not sigs:
        raise RuntimeError("No fold predictions produced; check models and data alignment.")
    all_sig = pd.concat(sigs).sort_index()
    if all_sig.index.has_duplicates:
        all_sig = all_sig[~all_sig.index.duplicated(keep="last")]

    print(f"[SIG]   {all_sig.index.min()} → {all_sig.index.max()}  rows={len(all_sig)}")

    # Portfolio metrics & trades computed on the TRIMMED price series
    metrics = pnl_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)

    trades_df = trades_from_signals(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        max_hold_bars=hold_bars,                 # bars, not hours
        high=df.get("high"),
        low=df.get("low"),
        entry_barrier_mult=barrier_daily_atr_mult,
        daily_atr=df_feat.get("atr_d1_14"),
    )

    print(
        f"[TRADES] entries: {trades_df['entry_time'].min() if not trades_df.empty else None} "
        f"→ {trades_df['entry_time'].max() if not trades_df.empty else None} | "
        f"exits: {trades_df['exit_time'].min() if not trades_df.empty else None} "
        f"→ {trades_df['exit_time'].max() if not trades_df.empty else None} | "
        f"n={len(trades_df)}"
    )

    if EXPORT_TRADES_CSV:
        trades_df.to_csv(EXPORT_TRADES_CSV, index=False)
        print(f"[SAVE] trades → {EXPORT_TRADES_CSV}")

    # Daily PnL/equity covering all days in the TRIMMED date range (even flat)
    daily_pnl = pnl_timeseries(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        resample="D",
        start_equity=1.0
    )
    daily_pnl.to_csv(DAILY_PNL_CSV)
    print(f"[SAVE] daily PnL → {DAILY_PNL_CSV}")

    summary = summarize_trades(trades_df)
    return {
        "metrics": metrics,
        "n_preds": int(all_sig.notna().sum()),
        "n_trades": int(len(trades_df)),
        "trade_summary": summary,
        "daily_pnl_csv": DAILY_PNL_CSV,
        "recent_start": recent_start,
    }


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"--data not found: {DATA_PATH}")
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"--models_dir not found: {MODELS_DIR}")

    out = predict_and_backtest(
        data_path=DATA_PATH,
        models_dir=MODELS_DIR,
        pmin=PMIN,
        margin=MARGIN,
        one_way_cost_bp=COST_BP,
        export_trades_csv=EXPORT_TRADES_CSV,
        hold_days=HOLD_DAYS,
        barrier_daily_atr_mult=BARRIER_DATR_MULT,
        recent_start=RECENT_START,   # <<< align with train_nn.py
    )
    print(out)
