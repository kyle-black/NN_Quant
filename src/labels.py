import numpy as np
import pandas as pd

def make_direction_label(df: pd.DataFrame, horizon: int = 10, thr: float = 0.0) -> pd.Series:
    """
    3-class label: -1, 0, +1 based on forward pct move over 'horizon'.
    IMPORTANT: this function returns a Series aligned to the CURRENT row,
    but it uses forward returns via shift(-horizon) to avoid leakage.
    """
    fwd = df['close'].shift(-horizon) / df['close'] - 1.0
    y = fwd.copy()
    y[:] = 1
    y[fwd > thr] = 2
    y[fwd < -thr] = 0
    # final alignment: last 'horizon' rows become NaN because we don't know future
    y.iloc[-horizon:] = np.nan
    return y
