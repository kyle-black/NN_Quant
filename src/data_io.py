# data_io.py
import pandas as pd

def load_forex_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Load FX CSV and return OHLC(V) with a clean, unique DateTimeIndex.
    We explicitly DROP vendor extras (label, change, vwap, etc.) so only numeric
    columns remain downstream.
    """
    df = pd.read_csv(path)

    # Identify timestamp column (date or timestamp)
    cols_lower = {c.lower(): c for c in df.columns}
    ts_col = None
    for candidate in ("timestamp", "date", "datetime"):
        if candidate in cols_lower:
            ts_col = cols_lower[candidate]
            break
    if ts_col is None:
        raise ValueError("CSV must have a 'timestamp' or 'date' column")

    # Parse to UTC datetime index
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col)

    # Normalize column names and keep ONLY ohlc + optional volume
    # Drop vendor-specific string fields like 'label', 'change', 'vwap', etc.
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            ren[c] = cl
    df = df.rename(columns=ren)

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()

    # Sort & dedupe index
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # Final numeric coercion & drop bad rows
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    return df
