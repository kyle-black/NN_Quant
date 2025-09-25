#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

from data_io import load_forex_csv
from features import add_basic_features
from pipelines import RollingScaler
from backtest import (
    softmax_to_dir_threshold,
    pnl_from_signals,
    trades_from_signals,
    summarize_trades,
)

from set_seed_all import seed_everything

SEED = 42
seed_everything(SEED, deterministic_tf=True)
#tf.random.set_seed(1234)
# =========================
# HARD-CODED PARAMETERS
# =========================
DATA_PATH   = "data/EURUSD_eod_2000_to_today.csv"   # daily file
MODELS_DIR  = "models/eurusd_nn"                    # folder with fold_*.keras + fold_*_feature_order.csv
PMIN        = 0.00                            # min prob to act
MARGIN      = 0.01                               # min edge over 2nd-best class (None to disable)
COST_BP     = 0.0                                   # one-way cost in basis points
HOLD_DAYS   = 21                                     # None for no time-exit; integer N => N bars if daily, 24*N bars if hourly
BARRIER_DATR_MULT = 3.0                             # None to disable; else k * daily ATR at entry
EXPORT_TRADES_CSV = "daily_export_trades_60_Min.csv"       # trades CSV (set None to skip writing)
# =========================


def _load_feature_cols(feat_path: str) -> list[str]:
    """
    Load saved feature order. Supports:
    - New format: single column with header 'feature'
    - Old format: headerless single column
    """
    try:
        return (
            pd.read_csv(feat_path)["feature"]
            .astype(str).str.strip().tolist()
        )
    except Exception:
        fc = (
            pd.read_csv(feat_path, header=None)
            .iloc[:, 0].astype(str).str.strip().tolist()
        )
        if fc in ([], ["0"], ["Unnamed: 0"], ["feature"]):
            raise RuntimeError(
                f"Feature list at {feat_path} looks invalid: {fc}. "
                "Retrain or re-save the feature order with a proper header."
            )
        return fc


def _infer_bars_per_day(index: pd.DatetimeIndex) -> int:
        """
        Infer how many bars per calendar day to convert HOLD_DAYS -> bars.
        Returns 1 for daily, 24 for hourly; falls back conservatively.
        """
        # Try pandas' inference first
        freq = pd.infer_freq(index)
        if freq:
            f = freq.upper()
            if "D" in f:
                return 1
            if "H" in f:
                return 24
        # Heuristic fallback: compare median spacing
        if len(index) >= 3:
            dt = index.to_series().diff().median()
            # ~1 day
            if pd.Timedelta(hours=12) < dt < pd.Timedelta(days=2):
                return 1
            # ~1 hour
            if pd.Timedelta(minutes=30) < dt < pd.Timedelta(hours=2):
                return 24
        # Default
        return 24


def predict_and_backtest(
    data_path: str,
    models_dir: str,
    pmin: float = 0.60,
    margin: float | None = 0.05,
    one_way_cost_bp: float = 0.5,
    export_trades_csv: str | None = None,
    hold_days: int | None = None,
    barrier_daily_atr_mult: float | None = None,
) -> dict:

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")

    def _fold_num(name: str) -> int:
        try:
            return int(name.split("_")[1].split(".")[0])
        except Exception:
            return 10**9

    files = sorted(keras_files, key=_fold_num)

    # Load data + features
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    print(f"[DATA]  {df.index.min()} → {df.index.max()}  rows={len(df)}")
    print(f"[FEAT]  {df_feat.index.min()} → {df_feat.index.max()}  rows={len(df_feat)}")

    # Proper OOS split size (e.g., 20% tail per fold)
    n = len(df_feat)
    test_size = 0.20
    test_n = int(max(1, n * test_size))
    n_folds = len(files)

    # Convert hold_days -> bars (auto daily vs hourly)
    if hold_days is not None:
        bars_per_day = _infer_bars_per_day(df_feat.index)
        hold_bars = int(hold_days * bars_per_day)
        print(f"[EXIT] HOLD_DAYS={hold_days} → bars_per_day={bars_per_day} → max_hold_bars={hold_bars}")
    else:
        hold_bars = None

    # Safety: only use barrier if daily ATR is present
    if barrier_daily_atr_mult is not None and "atr_d1_14" not in df_feat.columns:
        print("[WARN] barrier_daily_atr_mult set but 'atr_d1_14' not in features; disabling barrier exits.")
        barrier_daily_atr_mult = None

    sigs = []

    for f in files:
        fold = _fold_num(f)
        model = keras.models.load_model(os.path.join(models_dir, f))

        feat_path = os.path.join(models_dir, f"fold_{fold}_feature_order.csv")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Missing feature order file for {f}: {feat_path}")
        feature_cols = _load_feature_cols(feat_path)

        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            raise KeyError(
                f"Missing features in df_feat: {missing[:12]} ... "
                f"\nAvailable: {list(df_feat.columns)[:20]}..."
            )

        # Expanding-window style: each fold gets the same size tail, different start
        end_train = int((n - test_n) * (fold / n_folds))
        end_train = max(end_train, 1)
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))

        if len(te) == 0 or len(tr) < 10:
            # Skip pathological fold (too small train)
            print(f"[FOLD {fold}] skipped (train too small or no test).")
            continue

        print(f"[FOLD {fold}] test slice: {df_feat.index[te][0]} → {df_feat.index[te][-1]}  (len={len(te)})")

        X = df_feat[feature_cols].copy()
        scaler = RollingScaler()
        scaler.fit(X.iloc[tr])            # fit scaler on train only
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

    # Compute metrics (these functions handle 1-bar latency internally)
    metrics = pnl_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)

    trades_df = trades_from_signals(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        max_hold_bars=hold_bars,                 # <<< now bars, not hours
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

    if export_trades_csv:
        trades_df.to_csv(export_trades_csv, index=False)
        print(f"[SAVE] trades → {export_trades_csv}")

    summary = summarize_trades(trades_df)
    return {
        "metrics": metrics,
        "n_preds": int(all_sig.notna().sum()),
        "n_trades": int(len(trades_df)),
        "trade_summary": summary,
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
    )
    print(out)
