import os
import numpy as np
import pandas as pd
from tensorflow import keras

from data_io import load_forex_csv
from features import add_basic_features
from pipelines import RollingScaler
from backtest import softmax_to_dir, pnl_from_signals


def _load_feature_cols(feat_path: str) -> list[str]:
    """
    Load saved feature order. Supports:
    - New format: single column CSV with header 'feature'
    - Old format: headerless single column
    """
    try:
        return (
            pd.read_csv(feat_path)["feature"]
            .astype(str).str.strip().tolist()
        )
    except Exception:
        # Fallback to headerless
        fc = (
            pd.read_csv(feat_path, header=None)
            .iloc[:, 0].astype(str).str.strip().tolist()
        )
        # guard against junk
        if fc in ([], ["0"], ["Unnamed: 0"], ["feature"]):
            raise RuntimeError(
                f"Feature list at {feat_path} looks invalid: {fc}. "
                "Retrain or re-save the feature order with a proper header."
            )
        return fc


def predict_and_backtest(data_path: str, models_dir: str) -> dict:
    """
    Load each saved fold model, generate out-of-fold preds on its test slice,
    stitch together, and compute PnL metrics without leakage.
    """
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir not found: {models_dir}")

    # discover folds (expect names like: fold_1.keras)
    keras_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
    if not keras_files:
        raise FileNotFoundError(f"No .keras models found in {models_dir}")

    # sort by fold number if present; else lexicographic
    def _fold_num(name: str) -> int:
        # fold_12.keras -> 12
        try:
            return int(name.split("_")[1].split(".")[0])
        except Exception:
            return 10**9  # push odd names to the end
    files = sorted(keras_files, key=_fold_num)

    # load data + features
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    preds = []
    sigs = []

    n = len(df_feat)
    test_size = 0.2
    test_n = int(max(1, n * test_size))
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

        # validate features
       # missing = [c for c in feature_cols if c not in df_feat.columns]
       # if missing:
       #     raise KeyError(
        #        "Saved feature columns are missing from df_feat:\n"
        #        f"- missing: {missing}\n"
        #        f"- df_feat columns (sample): {list(df_feat.columns)[:15]}\n"
        #        f"- loaded feature_cols (sample): {feature_cols[:15]}"
        #    )

        # emulate training split math (approximation to the original expanding split)
        end_train = int((n - test_n) * (fold / n_folds))
        end_train = max(end_train, 1)  # safety
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))

        # scale on train only
        X = df_feat[feature_cols].copy()
        scaler = RollingScaler()
        scaler.fit(X.iloc[tr])
        X_te = scaler.transform(X.iloc[te])

        # predict → signals
        prob = model.predict(X_te, verbose=0)
        dir_sig = softmax_to_dir(prob)

        print(f'probs:{prob}')

        preds.append(prob)
        sigs.append(pd.Series(dir_sig, index=df_feat.index[te]))

    # stitch and backtest
    all_sig = pd.concat(sigs).sort_index()
    metrics = pnl_from_signals(df["close"], all_sig)
    return {"metrics": metrics, "n_preds": int(all_sig.notna().sum())}


if __name__ == "__main__":
    # Example CLI usage:
    # python predict_nn.py --data path/to/EURUSD.csv --models_dir models/eurusd_nn
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models_dir", required=True)
    args = p.parse_args()

    out = predict_and_backtest(args.data, args.models_dir)
    print(out)
