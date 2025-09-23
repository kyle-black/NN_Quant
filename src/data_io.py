import os


import pandas as pd

def load_forex_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load FX OHLCV CSV and return a DataFrame with a unique, sorted DatetimeIndex.
    - Accepts either a 'timestamp' column or an existing DatetimeIndex.
    - Normalizes columns to lower-case: open, high, low, close, volume (if present).
    - Ensures strictly increasing, unique index (drops duplicate timestamps, keep='last').
    """
    df = pd.read_csv(path)

    # Use timestamp column if present (case-insensitive), else assume DatetimeIndex
    cols_lower = {c.lower(): c for c in df.columns}
    if "timestamp" in cols_lower:
        ts_col = cols_lower["timestamp"]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        if tz and tz.upper() != "UTC":
            df[ts_col] = df[ts_col].dt.tz_convert(tz)
        df = df.set_index(ts_col)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provide a 'timestamp' column or a DatetimeIndex in the CSV.")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if tz and tz.upper() != "UTC":
            df.index = df.index.tz_convert(tz)

    # normalize column names to lower-case
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            ren[c] = cl
    df = df.rename(columns=ren)

    # basic sanity
    needed = {"open", "high", "low", "close"}
    have = set(c.lower() for c in df.columns)
    if not needed.issubset(have):
        raise ValueError(f"Missing columns: {sorted(needed - have)}")

    # sort and drop duplicate timestamps (keep the last occurrence)
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # drop obvious NaNs
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df
