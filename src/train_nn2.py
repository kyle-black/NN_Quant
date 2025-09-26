#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from tensorflow import keras

from set_seed_all import seed_everything
from data_io import load_forex_csv
from features import add_basic_features
from labels import make_direction_label_barrier_first_touch_days  # your barrier-first-touch label
from pipelines import RollingScaler, build_Xy
from timesplit import time_series_split
from model_keras import make_classifier

# ------------------------------
# Determinism
# ------------------------------
SEED = 42
seed_everything(SEED, deterministic_tf=True)

# ------------------------------
# HARD-CODED PARAMETERS
# ------------------------------
DATA_PATH       = "data/EURUSD_1h_2005-01-01_to_2025-09-23.csv"
OUT_DIR         = "models/eurusd_nn"
N_SPLITS        = 5
TEST_SIZE       = 0.20       # tail per fold (on the TRIMMED recent window)
EPOCHS          = 50
BATCH_SIZE      = 512

# LABEL / TARGET
HORIZON_BARS    = 21         # forward bars used by label (no leakage)
ATR_MULT        = 1.0        # barrier size in ATR units for label
ATR_COL_NAME    = "atr_14"   # feature column to use for barrier sizing at label time

# ------------- OPTION A -------------
# Train ONLY on data from this date onward (recent window)
RECENT_START    = "2017-01-01"  # change to taste: "2019-01-01", "2020-01-01", etc.
# -----------------------------------


def train_eurusd_recent_only(
    data_path: str,
    out_dir: str,
    recent_start: str = RECENT_START,
    horizon: int = HORIZON_BARS,
    atr_mult: float = ATR_MULT,
    atr_col: str = ATR_COL_NAME,
    n_splits: int = N_SPLITS,
    test_size: float = TEST_SIZE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:

    os.makedirs(out_dir, exist_ok=True)

    # 1) Load & feature-engineer
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    # 2) Trim to recent window (Option A)
    recent_start_ts = pd.Timestamp(recent_start, tz=df_feat.index.tz) if df_feat.index.tz is not None else pd.Timestamp(recent_start)
    df_feat = df_feat.loc[df_feat.index >= recent_start_ts].copy()
    if df_feat.empty or len(df_feat) < 1000:
        raise ValueError(f"After trimming to {recent_start}, not enough data to train (rows={len(df_feat)}).")

    # 3) Labels (first-touch barrier, no leakage; horizon forward)
    y = make_direction_label_barrier_first_touch_days(
        df_feat,
        days_ahead=horizon,
        atr_mult=atr_mult,
        use_daily_atr=True,     # e.g., "atr_14" or "atr_d1_14" depending on your features
    )

    # 4) Build X/y (drops warmup NaNs; aligns indices)
    X, y = build_Xy(df_feat, y)
    assert (X.index.equals(y.index)), "X and y index misaligned after build_Xy"

    # Guard: only numeric features
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise TypeError(f"Non-numeric columns found in features: {non_num}")

    X_np = X.values  # keep order for Keras

    # 5) CV over the TRIMMED recent window
    metrics = []
    fold_models = []
    for fold, (tr, te) in enumerate(
        time_series_split(len(X_np), n_splits=n_splits, test_size=test_size),
        start=1
    ):
        X_tr_df = X.iloc[tr]
        X_te_df = X.iloc[te]
        y_tr = y.iloc[tr].astype(int)
        y_te = y.iloc[te].astype(int)

        # Scale on train only
        scaler = RollingScaler()
        scaler.fit(X_tr_df)
        X_tr = scaler.transform(X_tr_df)
        X_te = scaler.transform(X_te_df)

        # Build model
        model = make_classifier(input_dim=X_tr.shape[1])

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]

        # IMPORTANT for determinism in time series
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=0,
            callbacks=cbs,
        )

        # Evaluate on held-out (recent-window tail)
        ev = model.evaluate(X_te, y_te, verbose=0)
        fold_metrics = {
            "fold": fold,
            "val_loss": float(ev[0]),
            "val_acc": float(ev[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_start": str(X_tr_df.index.min()),
            "train_end":   str(X_tr_df.index.max()),
            "test_start":  str(X_te_df.index.min()),
            "test_end":    str(X_te_df.index.max()),
        }
        metrics.append(fold_metrics)

        # Save model & feature order for this fold
        model_path = os.path.join(out_dir, f"fold_{fold}.keras")
        model.save(model_path)

        feat_csv = os.path.join(out_dir, f"fold_{fold}_feature_order.csv")
        pd.Series(list(X.columns), name="feature").to_csv(feat_csv, index=False)

        fold_models.append(model_path)

    # 6) Save a small manifest to document training window & params
    manifest = {
        "data_path": data_path,
        "trained_on_from": RECENT_START,
        "index_min": str(df_feat.index.min()),
        "index_max": str(df_feat.index.max()),
        "horizon_bars": horizon,
        "atr_mult": atr_mult,
        "atr_col": atr_col,
        "n_splits": n_splits,
        "test_size": test_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "fold_models": fold_models,
        "seed": SEED,
    }
    with open(os.path.join(out_dir, "manifest_recent.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "folds": metrics,
        "n_features": X.shape[1],
        "feature_cols": list(X.columns),
        "trained_on_from": RECENT_START,
        "index_min": str(df_feat.index.min()),
        "index_max": str(df_feat.index.max()),
        "manifest": os.path.join(out_dir, "manifest_recent.json"),
    }


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"--data not found: {DATA_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    out = train_eurusd_recent_only(
        data_path=DATA_PATH,
        out_dir=OUT_DIR,
        recent_start=RECENT_START,
        horizon=HORIZON_BARS,
        atr_mult=ATR_MULT,
        atr_col=ATR_COL_NAME,
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    print(out)
