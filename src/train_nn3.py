#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from set_seed_all import seed_everything
from data_io import load_forex_csv
from features import add_basic_features
from labels import make_direction_label_barrier_first_touch_days
from pipelines import RollingScaler, build_Xy
from timesplit import time_series_split
from model_keras import make_classifier
from backtest import pnl_from_signals, softmax_to_dir_threshold

# ============================================================
# Determinism
# ============================================================
SEED = 42
seed_everything(SEED, deterministic_tf=True)

# ============================================================
# HARD-CODED PARAMS (tweak as needed)
# ============================================================
DATA_PATH       = "data/EURUSD_1h_2005-01-01_to_2025-09-23.csv"
OUT_DIR         = "models/eurusd_nn"

# Option A: train only on data from this date forward
RECENT_START    = "2005-01-01"

# CV / training
N_SPLITS        = 5
TEST_SIZE       = 0.20
BATCH_SIZE      = 512
TARGET_UPDATES  = 5000     # used to auto-size epochs per fold
MIN_EPOCHS      = 15
MAX_EPOCHS      = 1000
LR              = 1e-3

# Labels
HORIZON_BARS    = 21
ATR_MULT        = 3
ATR_COL_NAME    = "atr_14"   # or "atr_d1_14" if you prefer daily ATR for the barrier

# Risk thresholds used during training callback logging
CALLBACK_PMIN   = 0.0
CALLBACK_MARGIN = 0.05

# Grid to tune thresholds *after* training (per fold, on that fold's val slice)
TUNE_P_GRID     = (0.00,0.33,0.55, 0.60, 0.65, 0.70)
TUNE_M_GRID     = (0.00, 0.02, 0.05, 0.08)

TB_ROOT         = "tb_logs"  # TensorBoard root folder
# ============================================================


# ------------------------------------------------------------
# Utility: choose epochs to target ~N optimizer updates
# ------------------------------------------------------------
def choose_epochs_for_updates(n_train_rows, batch_size, target_updates=5000,
                              min_epochs=15, max_epochs=100):
    steps_per_epoch = max(1, int(np.ceil(n_train_rows / batch_size)))
    epochs = int(np.clip(np.ceil(target_updates / steps_per_epoch), min_epochs, max_epochs))
    return epochs, steps_per_epoch


# ------------------------------------------------------------
# TensorBoard callback: log recent-slice risk metrics each epoch
# ------------------------------------------------------------
class RiskMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, X_val, idx_val, close_series,
                 one_way_cost_bp=0.5, pmin=0.60, margin=0.05, tag_prefix="recent"):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.X_val = X_val
        self.idx_val = idx_val  # DatetimeIndex aligned to X_val rows
        self.close = close_series  # full close Series
        self.cost = one_way_cost_bp
        self.pmin = pmin
        self.margin = margin
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, epoch, logs=None):
        prob = self.model.predict(self.X_val, verbose=0)
        sig = softmax_to_dir_threshold(prob, pmin=self.pmin, margin=self.margin)
        sig = pd.Series(sig, index=self.idx_val).sort_index()
        metrics = pnl_from_signals(self.close, sig, one_way_cost_bp=self.cost)
        sharpe = float(metrics.get("sharpe", np.nan))
        maxdd  = float(metrics.get("max_drawdown", np.nan))  # negative
        cagr   = float(metrics.get("cagr", np.nan))

        with self.file_writer.as_default():
            tf.summary.scalar(f"{self.tag_prefix}/sharpe", sharpe, step=epoch)
            tf.summary.scalar(f"{self.tag_prefix}/max_drawdown", maxdd, step=epoch)
            tf.summary.scalar(f"{self.tag_prefix}/cagr", cagr, step=epoch)

        # expose into Keras logs for monitoring
        if logs is not None:
            logs[f"val_{self.tag_prefix}_sharpe"] = sharpe
            logs[f"val_{self.tag_prefix}_maxdd"] = maxdd
            logs[f"val_{self.tag_prefix}_cagr"] = cagr


# ------------------------------------------------------------
# Simple threshold tuner on the validation slice
# ------------------------------------------------------------
def tune_thresholds_on_recent(close_series, probs, idx, cost_bp=0.5,
                              grid_p=TUNE_P_GRID, grid_m=TUNE_M_GRID):
    best = None
    for p in grid_p:
        for m in grid_m:
            sig = softmax_to_dir_threshold(probs, pmin=p, margin=m)
            sig = pd.Series(sig, index=idx).sort_index()
            met = pnl_from_signals(close_series, sig, one_way_cost_bp=cost_bp)
            sharpe = met.get("sharpe", 0.0)
            maxdd  = met.get("max_drawdown", 0.0)  # negative
            # Penalize DD worse than -10% (tweak to taste)
            score  = sharpe + 0.5 * min(0.0, 0.10 + maxdd)
            if (best is None) or (score > best[0]):
                best = (score, p, m, met)
    return {"score": best[0], "pmin": best[1], "margin": best[2], "metrics": best[3]}


# ------------------------------------------------------------
# Main training function
# ------------------------------------------------------------
def train_eurusd_recent_only(
    data_path: str = DATA_PATH,
    out_dir: str = OUT_DIR,
    recent_start: str = RECENT_START,
    horizon: int = HORIZON_BARS,
    atr_mult: float = ATR_MULT,
    atr_col: str = ATR_COL_NAME,
    n_splits: int = N_SPLITS,
    test_size: float = TEST_SIZE,
    batch_size: int = BATCH_SIZE,
    target_updates: int = TARGET_UPDATES,
    min_epochs: int = MIN_EPOCHS,
    max_epochs: int = MAX_EPOCHS,
    lr: float = LR,
) -> dict:

    os.makedirs(out_dir, exist_ok=True)

    # 1) Data & features
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    # 2) Recent-only trim
    rs_ts = (pd.Timestamp(recent_start, tz=df_feat.index.tz)
             if (recent_start and df_feat.index.tz is not None)
             else pd.Timestamp(recent_start)) if recent_start else None
    if rs_ts is not None:
        df_feat = df_feat.loc[df_feat.index >= rs_ts].copy()
        if len(df_feat) < 1000:
            raise ValueError(f"After trimming to {recent_start}, not enough data to train (rows={len(df_feat)}).")

    # 3) Labels (barrier first-touch, horizon forward)
    y = make_direction_label_barrier_first_touch_days(
        df_feat, days_ahead=horizon, atr_mult=atr_mult, use_daily_atr=True
    )

    # 4) Build X/y
    X, y = build_Xy(df_feat, y)
    assert X.index.equals(y.index), "X and y must be index-aligned after build_Xy"
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        raise TypeError(f"Non-numeric feature columns found: {non_num}")

    X_np = X.values

    # 5) CV on recent-only window
    fold_summaries = []
    fold_model_paths = []
    for fold, (tr, te) in enumerate(time_series_split(len(X_np), n_splits=n_splits, test_size=test_size), start=1):
        X_tr_df, X_te_df = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr].astype(int), y.iloc[te].astype(int)

        # scaler on train only
        scaler = RollingScaler()
        scaler.fit(X_tr_df)
        X_tr = scaler.transform(X_tr_df)
        X_te = scaler.transform(X_te_df)

        # model
        model = make_classifier(input_dim=X_tr.shape[1])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # plan epochs to hit target updates
        epochs, steps_per_epoch = choose_epochs_for_updates(
            n_train_rows=len(X_tr_df), batch_size=batch_size,
            target_updates=target_updates, min_epochs=min_epochs, max_epochs=max_epochs
        )

        # TensorBoard + risk metrics
        log_dir = os.path.join(TB_ROOT, f"fold_{fold}_{int(time.time())}")
        os.makedirs(log_dir, exist_ok=True)

        risk_cb = RiskMetricsCallback(
            log_dir=log_dir,
            X_val=X_te,
            idx_val=X_te_df.index,
            close_series=df["close"],         # full price series (original index)
            one_way_cost_bp=0.0,
            pmin=CALLBACK_PMIN,
            margin=CALLBACK_MARGIN,
            tag_prefix="recent"
        )

        cbs = [
            keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
            risk_cb,
            keras.callbacks.ReduceLROnPlateau(monitor="val_recent_sharpe", mode="max",
                                              factor=0.5, patience=3, min_lr=1e-5, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_recent_sharpe", mode="max",
                                          patience=6, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(
                os.path.join(out_dir, f"fold_{fold}.keras"),
                monitor="val_recent_sharpe", mode="max", save_best_only=True, verbose=1
            ),
        ]

        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,              # IMPORTANT for time series
            callbacks=cbs,
            verbose=0
        )

        # Best model already saved by ModelCheckpoint; still evaluate current weights:
        ev = model.evaluate(X_te, y_te, verbose=0)

        # Optional: tune thresholds on this fold's validation slice (using the *current* model predictions)
        prob_val = model.predict(X_te, verbose=0)
        tune = tune_thresholds_on_recent(
            close_series=df["close"],
            probs=prob_val,
            idx=X_te_df.index,
            cost_bp=0.0,
            grid_p=TUNE_P_GRID,
            grid_m=TUNE_M_GRID
        )

        # Save tuned thresholds per fold
        thr_path = os.path.join(out_dir, f"fold_{fold}_thresholds.json")
        with open(thr_path, "w") as f:
            json.dump({"pmin": tune["pmin"], "margin": tune["margin"], "score": tune["score"], "metrics": tune["metrics"]}, f, indent=2)

        # Save feature order used by this fold
        feat_csv = os.path.join(out_dir, f"fold_{fold}_feature_order.csv")
        pd.Series(list(X.columns), name="feature").to_csv(feat_csv, index=False)

        fold_summary = {
            "fold": fold,
            "val_loss": float(ev[0]),
            "val_acc": float(ev[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "train_start": str(X_tr_df.index.min()),
            "train_end":   str(X_tr_df.index.max()),
            "test_start":  str(X_te_df.index.min()),
            "test_end":    str(X_te_df.index.max()),
            "chosen_epochs": int(epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "tuned_pmin": float(tune["pmin"]),
            "tuned_margin": float(tune["margin"]),
            "tuned_score": float(tune["score"]),
        }
        fold_summaries.append(fold_summary)
        fold_model_paths.append(os.path.join(out_dir, f"fold_{fold}.keras"))

    # Manifest to document training
    manifest = {
        "data_path": DATA_PATH,
        "trained_on_from": RECENT_START,
        "index_min": str(X.index.min()),
        "index_max": str(X.index.max()),
        "horizon_bars": HORIZON_BARS,
        "atr_mult": ATR_MULT,
        "atr_col": ATR_COL_NAME,
        "n_splits": N_SPLITS,
        "test_size": TEST_SIZE,
        "batch_size": BATCH_SIZE,
        "target_updates": TARGET_UPDATES,
        "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "lr": LR,
        "seed": SEED,
        "fold_models": fold_model_paths,
    }
    with open(os.path.join(out_dir, "manifest_recent.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "folds": fold_summaries,
        "n_features": X.shape[1],
        "feature_cols": list(X.columns),
        "trained_on_from": RECENT_START,
        "manifest": os.path.join(out_dir, "manifest_recent.json"),
    }


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"--data not found: {DATA_PATH}")
    os.makedirs(OUT_DIR, exist_ok=True)

    out = train_eurusd_recent_only()
    print(out)

    print("\nTensorBoard tip:")
    print(f"  tensorboard --logdir {TB_ROOT} --port 6006")
