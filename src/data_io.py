import os
import pandas as pd

def load_forex_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load EURUSD data. Expected columns (case-insensitive at load):
    ['timestamp','open','high','low','close','volume'] or a DatetimeIndex.
    """
    df = pd.read_csv(path)
    if 'timestamp' in {c.lower() for c in df.columns}:
        ts_col = [c for c in df.columns if c.lower() == 'timestamp'][0]
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(tz)
        df = df.set_index(ts_col).sort_index()
    else:
        # assume index-like time already present
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Provide a timestamp column or DatetimeIndex.")
        df.index = df.index.tz_convert(tz) if df.index.tz is not None else df.index.tz_localize(tz)
        df = df.sort_index()

    # normalize column names
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('open','high','low','close','volume'):
            ren[c] = cl
    df = df.rename(columns=ren)

    needed = {'open','high','low','close'}
    if not needed.issubset(set(df.columns.str.lower())):
        raise ValueError(f"Missing columns: {needed - set(df.columns.str.lower())}")

    df = df.dropna().copy()
    return df
