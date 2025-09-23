import numpy as np
import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purely backward-looking features. No future info.
    """
    out = df.copy()
    # log returns
    out['ret_1'] = np.log(out['close']).diff()
    # rolling stats (use min_periods to avoid lookahead via dropna later)
    for w in (5, 10, 20):
        out[f'roll_mean_{w}'] = out['ret_1'].rolling(w, min_periods=w).mean()
        out[f'roll_std_{w}']  = out['ret_1'].rolling(w, min_periods=w).std()
    # momentum
    for k in (2, 5, 10):
        out[f'mom_{k}'] = out['close'].pct_change(k)
    # RSI (backward only)
    delta = out['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    for w in (14,):
        roll_up = up.rolling(w, min_periods=w).mean()
        roll_dn = down.rolling(w, min_periods=w).mean()
        rs = roll_up / (roll_dn.replace(0, np.nan))
        out[f'rsi_{w}'] = 100 - (100 / (1 + rs))
    # drop warmup
    out = out.dropna().copy()
    return out
