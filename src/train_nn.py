import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from tensorflow import keras

from data_io import load_forex_csv
from features import add_basic_features
#from labels import make_direction_label
from timesplit import time_series_split
from pipelines import RollingScaler, build_Xy
from model_keras import make_classifier
from labels import make_direction_label_barrier_first_touch 

def train_eurusd(
    data_path: str,
    horizon: int = 10,
    thr: float = 0.0,
    n_splits: int = 5,
    test_size: float = 0.2,
    epochs: int = 150,
    batch_size: int = 512,
    out_dir: str = "models/eurusd_nn"
) -> Dict[str, Any]:

    os.makedirs(out_dir, exist_ok=True)
    df = load_forex_csv(data_path)
    df_feat = add_basic_features(df)
   # y = make_direction_label(df_feat, horizon=horizon, thr=thr)
    y = make_direction_label_barrier_first_touch(df_feat, horizon=336, atr_mult=0.5, atr_col="atr_14")  # or "ATR_14")


    X, y = build_Xy(df_feat, y)
    X_np = X.values  # keep columns order

    metrics = []
    for fold, (tr, te) in enumerate(time_series_split(len(X_np), n_splits=n_splits, test_size=test_size), start=1):
        X_tr_df = X.iloc[tr]
        X_te_df = X.iloc[te]
        y_tr = y.iloc[tr].astype(int)
        y_te = y.iloc[te].astype(int)

        scaler = RollingScaler()
        scaler.fit(X_tr_df)
        X_tr = scaler.transform(X_tr_df)
        X_te = scaler.transform(X_te_df)

        model = make_classifier(input_dim=X_tr.shape[1])

        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]

        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=cbs
        )

        # evaluate on the held-out time tail
        ev = model.evaluate(X_te, y_te, verbose=0)
        fold_metrics = {
            "fold": fold,
            "val_loss": float(ev[0]),
            "val_acc": float(ev[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te))
        }
        metrics.append(fold_metrics)

        # save each fold’s best model and the feature order
        model.save(os.path.join(out_dir, f"fold_{fold}.keras"))
        #X.columns.to_series().to_csv(os.path.join(out_dir, f"fold_{fold}_feature_order.csv"), index=False)
        feat_csv = os.path.join(out_dir, f"fold_{fold}_feature_order.csv")
        pd.Series(list(X.columns), name="feature").to_csv(feat_csv, index=False)

    return {
        "folds": metrics,
        "n_features": X.shape[1],
        "feature_cols": list(X.columns)
    }

if __name__ == "__main__":
    # Example:
    # python -m publish_code.securities.forex.train_nn --data data/EURUSD.csv
    '''
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to EURUSD CSV")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--thr", type=float, default=0.0)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--out", default="models/eurusd_nn")
    args = p.parse_args()
    Pip Value = (0.0001 / Exchange Rate) * Lot Size
    '''

    data_path='data/EURUSD_1h_2005-01-01_to_2025-09-23.csv'
    horizon = 336
    thr =0.005
    splits = 5
    test_size = 0.20
    epochs =5
    batch_size= 256
    out = "models/eurusd_nn"
    models_dir="models/eurusd_nn"
    
    
    
    
    result = train_eurusd(
        data_path=data_path,
        horizon=horizon,
        thr=thr,
        n_splits=splits,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        out_dir=out
    )
    print(result)
