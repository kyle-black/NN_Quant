import numpy as np
import pandas as pd

def _pick_col_case_insensitive(df: pd.DataFrame, name: str) -> str:
    low = {c.lower(): c for c in df.columns}
    if name.lower() not in low:
        raise KeyError(f"Required column '{name}' not found. Available: {list(df.columns)}")
    return low[name.lower()]

def make_direction_label_barrier_first_touch(
    df: pd.DataFrame,
    horizon: int = 10,
    atr_mult: float = 1.0,
    atr_col: str = "atr_14",
) -> pd.Series:
    """
    3-class label based on which ATR barrier is touched FIRST within 'horizon' bars.
      +1 if first touch is upper barrier: close_t >= close_0 + atr_mult * ATR_0
      -1 if first touch is lower barrier: close_t <= close_0 - atr_mult * ATR_0
       0 if neither barrier touched within the horizon
    Last 'horizon' rows are NaN (unknown future).
    Requires columns: 'close','high','low', and atr_col.
    """
    close_col = _pick_col_case_insensitive(df, "close")
    high_col  = _pick_col_case_insensitive(df, "high")
    low_col   = _pick_col_case_insensitive(df, "low")
    atr_col   = _pick_col_case_insensitive(df, atr_col)

    close = df[close_col].to_numpy(dtype=float)
    high  = df[high_col].to_numpy(dtype=float)
    low   = df[low_col].to_numpy(dtype=float)
    atr   = df[atr_col].to_numpy(dtype=float)

    n = len(df)
    y = np.full(n, np.nan, dtype=float)

    for i in range(0, n - horizon):
        c0 = close[i]
        a0 = atr[i]
        up = c0 + atr_mult * a0
        dn = c0 - atr_mult * a0

        j0, j1 = i + 1, i + horizon + 1
        hi_win = high[j0:j1]
        lo_win = low[j0:j1]

        hit_up = hi_win >= up
        hit_dn = lo_win <= dn

        first_up = np.argmax(hit_up) if hit_up.any() else 10**9
        first_dn = np.argmax(hit_dn) if hit_dn.any() else 10**9

        if first_up == 10**9 and first_dn == 10**9:
            y[i] = 1.0 #neutral
        elif first_up < first_dn:
            y[i] = 2.0 # up
        elif first_dn < first_up:
            y[i] = 0.0 #down
        else:
            y[i] = 1.0

    return pd.Series(y, index=df.index, name="label_barrier_first_touch")
