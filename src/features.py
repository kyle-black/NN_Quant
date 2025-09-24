import numpy as np
import pandas as pd

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-only features on 1h bars + a D1 ATR we can use for day-scale barriers.
    Requires columns: open, high, low, close (case-insensitive).
    """
    out = df.copy()
    # normalize cols
    colmap = {c.lower(): c for c in out.columns}
    for k in ("open","high","low","close"):
        if k not in colmap: raise KeyError(f"Missing column '{k}' in data")
    O,H,L,C = [colmap[x] for x in ("open","high","low","close")]

    # 1h log return
    out["ret_1"] = np.log(out[C]).diff()

    # Rolling stats on 1h
    for w in (5,10,20,50,100):
        out[f"roll_mean_{w}"] = out["ret_1"].rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"]  = out["ret_1"].rolling(w, min_periods=w).std()

    # Momentum windows
    for k in (2,5,10,24):
        out[f"mom_{k}"] = out[C].pct_change(k)

    # RSI(14) (Wilder)
    delta = out[C].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_dn = down.rolling(14, min_periods=14).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = _ema(out[C], 12); ema26 = _ema(out[C], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd"] = macd; out["macd_signal"] = signal; out["macd_hist"] = macd - signal

    # ATR(14) on 1h (Wilder-like using rolling mean of TR)
    prev_close = out[C].shift(1)
    tr1 = (out[H] - out[L]).abs()
    tr2 = (out[H] - prev_close).abs()
    tr3 = (out[L] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()

    # ---- NEW: Daily ATR(14) (computed on resampled D1 bars) ----
    d1 = out[[O,H,L,C]].resample("1D").agg({O:"first", H:"max", L:"min", C:"last"}).dropna()
    prev_close_d1 = d1[C].shift(1)
    tr_d1 = pd.concat([
        (d1[H]-d1[L]).abs(),
        (d1[H]-prev_close_d1).abs(),
        (d1[L]-prev_close_d1).abs()
    ], axis=1).max(axis=1)
    atr_d1_14 = tr_d1.rolling(14, min_periods=14).mean()
    # bring back to 1h index (forward-fill today’s ATR across the day)
    out["atr_d1_14"] = atr_d1_14.reindex(out.index, method="ffill")

    # Drop feature warmup rows
    out = out.dropna().copy()
    return out
