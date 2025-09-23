import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

class RollingScaler:
    """
    Fit ONLY on train indices, transform train and test separately.
    Prevents leakage from future distribution shift.
    """
    def __init__(self):
        self.scaler = None

    def fit(self, X: pd.DataFrame):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(X.values)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Call fit before transform.")
        return self.scaler.transform(X.values)

def build_Xy(df_feat: pd.DataFrame, y: pd.Series, feature_cols=None) -> Tuple[pd.DataFrame, pd.Series]:
    df = df_feat.copy()
    if feature_cols is None:
        # everything except price columns & y
        drop_cols = {'open','high','low','close','volume'}
        feature_cols = [c for c in df.columns if c not in drop_cols]
    # align
    y = y.reindex(df.index)
    mask = y.notna()
    return df.loc[mask, feature_cols], y.loc[mask]
