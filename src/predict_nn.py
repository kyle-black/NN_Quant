import os
import numpy as np
import pandas as pd
from tensorflow import keras

from .data_io import load_forex_csv
from .features import add_basic_features
from .pipelines import RollingScaler, build_Xy
from .backtest import softmax_to_dir, pnl_from_signals

def predict_and_backtest(data_path: str, models_dir: str) -> dict:
    """
    Load each saved fold model, generate out-of-fold preds on its test slice,
    stitch together, and compute PnL metrics without leakage.
    """
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)

    # collect fold pieces
    preds = []
    sigs = []
    idxs = []

    # discover folds
    files = sorted([f for f in os.listdir(models_dir) if f.endswith(".keras")])
    for f in files:
        fold = int(f.split("_")[1].split(".")[0])
        model = keras.models.load_model(os.path.join(models_dir, f))
        feat_path = os.path.join(models_dir, f"fold_{fold}_feature_order.csv")
        feature_cols = pd.read_csv(feat_path, header=None).iloc[:,0].tolist()

        # emulate the same split used during training:
        n = len(df_feat)
        test_size = 0.2
        test_n = int(max(1, n * test_size))
        end_train = int((n - test_n) * (fold / len(files)))
        tr = np.arange(0, end_train)
        te = np.arange(end_train, min(end_train + test_n, n))

        X = df_feat[feature_cols].copy()
        scaler = RollingScaler()
        scaler.fit(X.iloc[tr])
        X_te = scaler.transform(X.iloc[te])

        prob = model.predict(X_te, verbose=0)
        dir_sig = softmax_to_dir(prob)

        preds.append(prob)
        sigs.append(pd.Series(dir_sig, index=df_feat.index[te]))
        idxs.append(df_feat.index[te])

    all_sig = pd.concat(sigs).sort_index()
    metrics = pnl_from_signals(df['close'], all_sig)
    return {"metrics": metrics, "n_preds": int(all_sig.notna().sum())}
