import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    # Exponential moving average (no lookahead)
    return s.ewm(span=span, adjust=False).mean()

def _wilder_rma(s: pd.Series, length: int) -> pd.Series:
    # Wilder's "RMA" via ewm with alpha=1/length
    return s.ewm(alpha=1.0/length, adjust=False).mean()

def _rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder_rma(gain, length)
    avg_loss = _wilder_rma(loss, length)
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist
    })

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = _wilder_rma(tr, length)
    return atr

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-looking features only (no future info).
    Requires columns: 'close' (for all), and 'high','low' for ATR.
    """
    out = df.copy()

    # --- Base returns / stats ---
    out['ret_1'] = np.log(out['close']).diff()

    for w in (5, 10, 20):
        out[f'roll_mean_{w}'] = out['ret_1'].rolling(w, min_periods=w).mean()
        out[f'roll_std_{w}']  = out['ret_1'].rolling(w, min_periods=w).std()

    for k in (2, 5, 10):
        out[f'mom_{k}'] = out['close'].pct_change(k)

    # --- RSI (Wilder) ---
    out['rsi_14'] = _rsi_wilder(out['close'], length=14)

    # --- MACD (12,26,9) ---
    macd_df = _macd(out['close'], fast=12, slow=26, signal=9)
    out = out.join(macd_df)

    # --- ATR (14) ---
    if not {'high','low','close'}.issubset({c.lower() for c in out.columns}):
        raise ValueError("ATR requires 'high','low','close' columns.")
    out['atr_14'] = _atr(out['high'], out['low'], out['close'], length=14)

    # Drop warmup NaNs created by rolling/ewm; keep only fully-formed rows
    out = out.dropna().copy()
    return out
