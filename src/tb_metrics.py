# tb_metrics.py
import os, numpy as np, tensorflow as tf
import pandas as pd

from backtest import pnl_from_signals, softmax_to_dir_threshold

class RiskMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, X_val, y_val, idx_val, close_series,
                 one_way_cost_bp=0.5, pmin=0.60, margin=0.05, tag_prefix="recent"):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.X_val = X_val
        self.y_val = y_val
        self.idx_val = idx_val  # DatetimeIndex aligned to X_val rows
        self.close = close_series  # full close Series
        self.cost = one_way_cost_bp
        self.pmin = pmin
        self.margin = margin
        self.tag_prefix = tag_prefix

    def on_epoch_end(self, epoch, logs=None):
        # 1) Predict probs
        prob = self.model.predict(self.X_val, verbose=0)
        # 2) Convert to directional signals
        sig = softmax_to_dir_threshold(prob, pmin=self.pmin, margin=self.margin)
        sig = pd.Series(sig, index=self.idx_val).sort_index()
        # 3) Compute PnL metrics on the matching close series
        metrics = pnl_from_signals(self.close, sig, one_way_cost_bp=self.cost)
        sharpe = float(metrics.get("sharpe", np.nan))
        maxdd  = float(metrics.get("max_drawdown", np.nan))  # expect negative values
        cagr   = float(metrics.get("cagr", np.nan))

        # 4) Log to TensorBoard
        with self.file_writer.as_default():
            tf.summary.scalar(f"{self.tag_prefix}/sharpe", sharpe, step=epoch)
            tf.summary.scalar(f"{self.tag_prefix}/max_drawdown", maxdd, step=epoch)
            tf.summary.scalar(f"{self.tag_prefix}/cagr", cagr, step=epoch)

        # Also put them into Keras logs so you can monitor/stop on them
        if logs is not None:
            logs[f"val_{self.tag_prefix}_sharpe"] = sharpe
            logs[f"val_{self.tag_prefix}_maxdd"] = maxdd
            logs[f"val_{self.tag_prefix}_cagr"] = cagr
