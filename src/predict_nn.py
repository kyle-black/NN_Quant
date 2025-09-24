#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from data_io import load_forex_csv
from features import add_basic_features
from pipelines import RollingScaler
from backtest import (
    softmax_to_dir_threshold,
    pnl_from_signals,
    trades_from_signals,
    summarize_trades,
)

# =========================
# HARD-CODED PARAMETERS
# =========================
DATA_PATH   = "data/EURUSD_eod_2000_to_today.csv"
MODELS_DIR  = "models/eurusd_nn"        # folder with fold_*.keras + fold_*_feature_order.csv
PMIN        = 0.00                    # min prob to act
MARGIN      = 0.01                     # min edge over 2nd-best class (None to disable)
COST_BP     = 0.0                       # one-way cost in basis points
HOLD_DAYS   = None                        # None for no time-exit; integer N => force exit after 24*N bars
BARRIER_DATR_MULT = 1.0                 # None to disable triple-barrier; else k * daily ATR at entry
EXPORT_TRADES_CSV = "trades_eurusd_pred.csv"  # output trades CSV (set None to skip writing)
# =========================


def _load_feature_cols(feat_path: str) -> list[str]:
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


def predict_and_backtest(
    data_path: str,
    models_dir: str,
    pmin: float = 0.60,
    margin: float | None = 0.05,
    one_way_cost_bp: float = 0.5,
    export_trades_csv: str ='trade_output_daily.csv',
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

    sigs = []
    n = len(df_feat)
    test_size = 0.00
    test_n = int(max(1, n ))
    n_folds = len(files)

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

        end_train = int((n - test_n) * (fold / n_folds))
        end_train = max(end_train, 1)
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))

        if len(te) == 0:
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

    # Convert hold_days → hours for 1h bars
    hold_hours = None if hold_days is None else int(hold_days * 24)

    # Metrics (latency and de-dupe handled inside)
    metrics = pnl_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)

    trades_df = trades_from_signals(
        close=df["close"],
        signals=all_sig,
        one_way_cost_bp=one_way_cost_bp,
        max_hold_hours=hold_hours,
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
        trades_df.to_csv('daily_export_trades.csv', index=False)
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
