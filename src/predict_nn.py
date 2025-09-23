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


def predict_and_backtest(
    data_path: str,
    models_dir: str,
    pmin: float = 0.60,
    margin: float | None = 0.05,
    one_way_cost_bp: float = 0.5,
    export_trades_csv: str | None = None,
) -> dict:
    """
    Stitch predictions from saved fold models, apply probability threshold,
    compute PnL, and parse trades.

    Args:
        data_path: path to CSV (timestamp index or 'timestamp' column)
        models_dir: directory containing fold_{k}.keras and fold_{k}_feature_order.csv
        pmin: min prob to act (else abstain = 0)
        margin: optional min gap between top2 probs
        one_way_cost_bp: transaction costs per side in bps
        export_trades_csv: if provided, save trades to this CSV

    Returns:
        dict with metrics, trade summary, counts, etc.
    """
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

    # load data & features
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    sigs = []
    n = len(df_feat)
    test_size = 0.2
    test_n = int(max(n+15000, n ))
    n_folds = len(files)

    for f in files:
        fold = _fold_num(f)
        model = keras.models.load_model(os.path.join(models_dir, f))

        feat_path = os.path.join(models_dir, f"fold_{fold}_feature_order.csv")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Missing feature order file for {f}: {feat_path}"
            )
        feature_cols = _load_feature_cols(feat_path)

        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            raise KeyError(
                "Saved feature columns are missing from df_feat:\n"
                f"- missing: {missing}\n"
                f"- df_feat columns (sample): {list(df_feat.columns)[:15]}\n"
                f"- loaded feature_cols (sample): {feature_cols[:15]}"
            )

        # emulate training split (expanding window approx)
        end_train = int((n - test_n) * (fold / n_folds))
        end_train = max(end_train, 1)
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))

        # scale on train only
        X = df_feat[feature_cols].copy()
        scaler = RollingScaler()
        scaler.fit(X.iloc[tr])
        X_te = scaler.transform(X.iloc[te])

        # predict -> thresholded signals
        prob = model.predict(X_te, verbose=0)
        dir_sig = softmax_to_dir_threshold(prob, pmin=pmin, margin=margin)
        sigs.append(pd.Series(dir_sig, index=df_feat.index[te]))

    # stitch signals, make duplicate-safe and sorted
    all_sig = pd.concat(sigs)
    all_sig = all_sig.sort_index()
    if all_sig.index.has_duplicates:
        all_sig = all_sig[~all_sig.index.duplicated(keep="last")]

    # PnL & trades (these functions also align & de-dup internally)
    metrics = pnl_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)
    trades_df = trades_from_signals(df["close"], all_sig, one_way_cost_bp=one_way_cost_bp)
    trades_df.to_csv('trades_df.csv')
    summary = summarize_trades(trades_df)

    if export_trades_csv:
        trades_df.to_csv(export_trades_csv, index=False)

    return {
        "metrics": metrics,
        "n_preds": int(all_sig.notna().sum()),
        "n_trades": int(len(trades_df)),
        "trade_summary": summary,
        # "trades": trades_df.to_dict(orient="records"),  # enable if you want raw trades in-memory
    }


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("NNQ_DATA", "data/EURUSD_1h_2005-01-01_to_2025-09-23.csv"))
    p.add_argument("--models_dir", default=os.environ.get("NNQ_MODELS", "models/eurusd_nn"))
    p.add_argument("--pmin", type=float, default=0.30)
    p.add_argument("--margin", type=float, default=0.00)
    p.add_argument("--cost_bp", type=float, default=0.0)
    p.add_argument("--export_trades_csv", default=None)
    args = p.parse_args()

    # quick guard so we fail early if files don’t exist
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"--data not found: {args.data}")
    if not os.path.isdir(args.models_dir):
        raise FileNotFoundError(f"--models_dir not found: {args.models_dir}")

    out = predict_and_backtest(
        data_path=args.data,
        models_dir=args.models_dir,
        pmin=args.pmin,
        margin=args.margin,
        one_way_cost_bp=args.cost_bp,
        export_trades_csv=args.export_trades_csv,
    )
    print(out)
