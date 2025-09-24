#!/usr/bin/env python3
# Fetch EURUSD daily history back to 2000 with robust endpoint fallbacks.

import json, time, ssl, certifi
from datetime import datetime, timedelta
from urllib.parse import urlencode
try:
    from urllib.request import urlopen, Request
except ImportError:
    from urllib2 import urlopen, Request
import pandas as pd

APIKEY      = "NLwwRkkEzJIVdRCiZqUBlIMa4z6M8mH3"
SYMBOL      = "EURUSD"
START_DATE  = "2000-01-01"
END_DATE    = None        # None -> today UTC
CHUNK_DAYS  = 180
PAUSE_SEC   = 0.35
MAX_RETRIES = 4
OUT_PATH    = "EURUSD_eod_2000_to_today.csv"

# Candidate bases (try in order)
BASES_PATH_STYLE = [
    "https://financialmodelingprep.com/api/v3/historical-price-full",   # preferred
    "https://financialmodelingprep.com/stable/historical-price-full",   # fallback
]
# “eod/full?symbol=” style fallback
EOD_FULL = "https://financialmodelingprep.com/stable/historical-price-eod/full"

CTX = ssl.create_default_context(cafile=certifi.where())

def http_get_json(url: str, timeout: int = 30):
    req = Request(url, headers={"User-Agent": "nn-quant-fetcher/1.0"})
    with urlopen(req, context=CTX, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)

def normalize_to_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = pd.DataFrame(rows)
    lower = {c.lower(): c for c in df.columns}
    def pick(name): return lower.get(name, None)
    cols = {"date": pick("date"),
            "open": pick("open"),
            "high": pick("high"),
            "low":  pick("low"),
            "close": pick("close") or pick("adjclose"),
            "volume": pick("volume")}
    cols = {k:v for k,v in cols.items() if v}
    df = df.rename(columns={v:k for k,v in cols.items()})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_window_path_style(symbol: str, apikey: str, start: str, end: str):
    params = {"from": start, "to": end, "apikey": apikey}
    last_err = None
    for base in BASES_PATH_STYLE:
        url = f"{base}/{symbol}?{urlencode(params)}"
        try:
            j = http_get_json(url)
            if isinstance(j, dict) and "historical" in j:
                return j["historical"]
            if isinstance(j, list):
                return j
        except Exception as e:
            last_err = e
            continue
    # final fallback: eod/full?symbol=...
    try:
        url = f"{EOD_FULL}?{urlencode({'symbol': symbol, 'from': start, 'to': end, 'apikey': apikey})}"
        j = http_get_json(url)
        if isinstance(j, dict) and "historical" in j:
            return j["historical"]
        if isinstance(j, list):
            return j
    except Exception as e:
        last_err = e
    raise last_err or RuntimeError("All endpoints failed for window fetch")

def fetch_paged(symbol: str, apikey: str,
                start_date: str, end_date: str | None,
                chunk_days: int, pause_sec: float, max_retries: int) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.utcnow().date().isoformat()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("start_date must be <= end_date")

    all_rows = []
    cur = start_dt
    while cur <= end_dt:
        win_end = min(end_dt, cur + timedelta(days=chunk_days-1))
        s, e = cur.isoformat(), win_end.isoformat()

        for attempt in range(1, max_retries+1):
            try:
                rows = fetch_window_path_style(symbol, apikey, s, e)
                all_rows.extend(rows)
                break
            except Exception:
                if attempt >= max_retries:
                    raise
                time.sleep(pause_sec * (2 ** (attempt-1)))
        time.sleep(pause_sec)
        cur = win_end + timedelta(days=1)

    return normalize_to_df(all_rows)

def main():
    df = fetch_paged(
        symbol=SYMBOL, apikey=APIKEY,
        start_date=START_DATE, end_date=END_DATE,
        chunk_days=CHUNK_DAYS, pause_sec=PAUSE_SEC, max_retries=MAX_RETRIES
    )
    if df.empty:
        print("No data returned.")
        return
    print(f"Fetched {len(df):,} rows from {df['date'].min().date()} to {df['date'].max().date()}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
