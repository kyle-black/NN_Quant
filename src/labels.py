import numpy as np
import pandas as pd

def _pick(df: pd.DataFrame, name: str) -> str:
    low = {c.lower(): c for c in df.columns}
    if name.lower() not in low:
        raise KeyError(f"Missing '{name}' in DataFrame.")
    return low[name.lower()]

def make_direction_label_barrier_first_touch_days(
    df: pd.DataFrame,
    days_ahead: int = 3,          # e.g., 3 trading days ahead on 1h bars
    atr_mult: float = 1.0,
    use_daily_atr: bool = True,   # True: use atr_d1_14; False: sqrt-time scale hourly atr_14
    base_hour_horizon: int = 14   # for sqrt-time scaling if use_daily_atr=False
) -> pd.Series:
    """
    3-class label via first-touch barrier within 'days_ahead' days using 1h bars.
      +1 if upper barrier touched first, -1 if lower, else 0 by the deadline.
    Barriers:
      - If use_daily_atr: close0 +/- atr_mult * atr_d1_14[i]
      - Else:             close0 +/- atr_mult * atr_14[i] * sqrt(H / base_hour_horizon)
    """
    C = _pick(df, "close"); H = _pick(df, "high"); L = _pick(df, "low")

    # convert days → hours (on 1h bars). Using 24 per calendar day.
    horizon = int(days_ahead * 24)
    if horizon <= 0:
        raise ValueError("days_ahead must be >= 1")

    close = df[C].to_numpy(float)
    high  = df[H].to_numpy(float)
    low   = df[L].to_numpy(float)

    if use_daily_atr:
        atrcol = _pick(df, "atr_d1_14")
        dist = atr_mult * df[atrcol].to_numpy(float)
    else:
        atrcol = _pick(df, "atr_14")
        dist = atr_mult * df[atrcol].to_numpy(float) * np.sqrt(horizon / max(1, base_hour_horizon))

    n = len(df)
    y = np.full(n, np.nan, float)

    for i in range(0, n - horizon):
        d = dist[i]
        if not np.isfinite(d):
            continue
        up = close[i] + d
        dn = close[i] - d

        j0, j1 = i + 1, i + horizon + 1
        hit_up = (high[j0:j1] >= up)
        hit_dn = (low[j0:j1]  <= dn)

        first_up = np.argmax(hit_up) if hit_up.any() else 10**9
        first_dn = np.argmax(hit_dn) if hit_dn.any() else 10**9

        if first_up == 10**9 and first_dn == 10**9:
            y[i] = 1.0 #neutral
        elif first_up < first_dn:
            y[i] = 2.0 #up
        elif first_dn < first_up:
            y[i] = 0.0 #down
        else:
            y[i] = 1.0

    # last horizon rows are unknown by design
    return pd.Series(y, index=df.index, name=f"label_ft_{days_ahead}d")
