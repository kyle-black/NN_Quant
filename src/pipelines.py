import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler

# pipelines.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class RollingScaler:
    """
    Thin wrapper over StandardScaler that ignores non-numeric columns and
    preserves column order.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number]).copy()
        self.columns_ = list(Xn.columns)
        self.scaler.fit(Xn.values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # align to columns seen in fit; missing become NaN then filled with 0 before scaling
        Xn = X[self.columns_].copy()
        Xn = Xn.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Z = self.scaler.transform(Xn.values)
        out = pd.DataFrame(Z, index=X.index, columns=self.columns_)
        return out


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
