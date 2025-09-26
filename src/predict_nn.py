#!/usr/bin/env python3
# predict_nn.py
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from set_seed_all import seed_everything
from data_io import load_forex_csv
from features import add_basic_features
from pipelines import RollingScaler
from backtest import (
    softmax_to_dir_longshort_gap,  # NEW decision rule
    confidence_scaled_mult,        # NEW helper to size barrier per entry
    pnl_from_signals,
    trades_from_signals,
    summarize_trades,
    pnl_timeseries,
)

# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
SEED = 42
seed_everything(SEED, deterministic_tf=True)

# =========================
# HARD-CODED PARAMETERS
# =========================
DATA_PATH   = "data/EURUSD_eod_2000_to_today.csv"   # daily by default
MODELS_DIR  = "models/eurusd_nn"                    # contains fold_*.keras + fold_*_feature_order.csv

# Decision rule (Long–Short gap)
PMIN_LS     = 0.0   # min(max(pL,pS)) to consider trading
GAP         = 0.20   # required |pL - pS|

# Costs / exits
COST_BP     = 0.0
HOLD_DAYS   = 30      # N bars if daily, 24*N if hourly

# Barrier sizing
BASE_K      = 4.0    # base ATR multiple (daily ATR)
K_MIN       = 0.75
K_MAX       = 3.0
USE_CONFIDENCE_SCALED_BARRIER = False  # if True, scale k by L–S confidence

# Outputs
EXPORT_TRADES_CSV = "daily_export_trades.csv"
DAILY_PNL_CSV     = "daily_pnl_timeseries.csv"
# =========================


def _load_feature_cols(feat_path: str) -> list[str]:
    try:
        return pd.read_csv(feat_path)["feature"].astype(str).str.strip().tolist()
    except Exception:
        fc = pd.read_csv(feat_path, header=None).iloc[:, 0].astype(str).str.strip().tolist()
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


def predict_and_backtest(
    data_path: str,
    models_dir: str,
    pmin_ls: float = PMIN_LS,
    gap: float = GAP,
    one_way_cost_bp: float = COST_BP,
    export_trades_csv: str | None = EXPORT_TRADES_CSV,
    hold_days: int | None = HOLD_DAYS,
    base_k: float = BASE_K,
    k_min: float = K_MIN,
    k_max: float = K_MAX,
    use_conf_scaled_barrier: bool = USE_CONFIDENCE_SCALED_BARRIER,
) -> dict:

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")

    def _fold_num(name: str) -> int:
        try: return int(name.split("_")[1].split(".")[0])
        except Exception: return 10**9

    files = sorted(keras_files, key=_fold_num)

    # Load data & features
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    print(f"[DATA]  {df.index.min()} → {df.index.max()}  rows={len(df)}")
    print(f"[FEAT]  {df_feat.index.min()} → {df_feat.index.max()}  rows={len(df_feat)}")

    # OOS slice size per fold
    n = len(df_feat)
    test_size = 0.20
    test_n = int(max(1, n * test_size))
    n_folds = len(files)

    # Convert hold_days -> bars (daily vs hourly)
    if hold_days is not None:
        bars_per_day = _infer_bars_per_day(df_feat.index)
        hold_bars = int(hold_days * bars_per_day)
        print(f"[EXIT] HOLD_DAYS={hold_days} → bars_per_day={bars_per_day} → max_hold_bars={hold_bars}")
    else:
        hold_bars = None

    # Barrier guard
    if "atr_d1_14" not in df_feat.columns:
        print("[WARN] 'atr_d1_14' not in features; barrier exits disabled.")
        base_k = None
        use_conf_scaled_barrier = False

    sigs = []
    k_series_pieces = []  # NEW: per-entry k Series aligned to test index

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
                "Missing features in df_feat:\n"
                f"- missing: {missing}\n"
                f"- available sample: {list(df_feat.columns)[:15]}"
            )

        # Expanding-window split within the (full or trimmed) features
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

        # Predict probs -> L–S gap signal
        prob = model.predict(X_te, verbose=0)  # (N,3) for [short, neutral, long]
        dir_sig = softmax_to_dir_longshort_gap(prob, pmin_ls=pmin_ls, gap=gap)

        sigs.append(pd.Series(dir_sig, index=df_feat.index[te]))

        # Optional: per-entry k from confidence
        if use_conf_scaled_barrier and base_k is not None:
            pS = prob[:, 0]
            pL = prob[:, 2]
            kvals = [
                confidence_scaled_mult(pl, ps, base_k=base_k, gap=gap, k_min=k_min, k_max=k_max)
                for ps, pl in zip(pS, pL)
            ]
            k_series_pieces.append(pd.Series(kvals, index=df_feat.index[te]))

    if not sigs:
        raise RuntimeError("No fold predictions produced; check models and data alignment.")

    all_sig = pd.concat(sigs).sort_index()
    if all_sig.index.has_duplicates:
        all_sig = all_sig[~all_sig.index.duplicated(keep="last")]
    print(f"[SIG]   {all_sig.index.min()} → {all_sig.index.max()}  rows={len(all_sig)}")

    # Merge per-entry k (optional)
    per_entry_mult = None
    if use_conf_scaled_barrier and base_k is not None and k_series_pieces:
        per_entry_mult = pd.concat(k_series_pieces).sort_index()
        if per_entry_mult.index.has_duplicates:
            per_entry_mult = per_entry_mult[~per_entry_mult.index.duplicated(keep="last")]

    # Portfolio metrics on prices (1-bar latency inside)
    metrics = pnl_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)

    # Trades with optional barrier & per-entry scaling
    trades_df = trades_from_signals(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        max_hold_bars=hold_bars,
        high=df.get("high"),
        low=df.get("low"),
        entry_barrier_mult=base_k,                # base k if per-entry not provided
        daily_atr=df_feat.get("atr_d1_14"),
        per_entry_mult=per_entry_mult,            # NEW: confidence-scaled k per entry
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

    daily_pnl = pnl_timeseries(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        resample="D",
        start_equity=1.0,
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
    }


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"--data not found: {DATA_PATH}")
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"--models_dir not found: {MODELS_DIR}")

    out = predict_and_backtest(
        data_path=DATA_PATH,
        models_dir=MODELS_DIR,
    )
    print(out)
