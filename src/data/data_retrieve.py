#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    # Python 3.x
    from urllib.request import urlopen
    from urllib.parse import urlencode
except ImportError:
    # Python 2.x (if you really need it)
    from urllib2 import urlopen
    from urllib import urlencode

import certifi
import json
import time
import csv
from datetime import datetime, timedelta

API_KEY = "NLwwRkkEzJIVdRCiZqUBlIMa4z6M8mH3"
SYMBOL = "EURUSD"
BASE = "https://financialmodelingprep.com/stable/historical-chart/1hour"

START_DATE = "2005-01-01"   # earliest you want
END_DATE = "2025-09-23"            # None = today; or "2025-09-23"
DAYS_PER_CHUNK = 30         # ~30-day windows
SLEEP_BETWEEN_CALLS = 0.25  # tweak if you hit 429s
MAX_RETRIES = 3
BACKOFF = 1.7               # exponential backoff factor
OUT_CSV = "{}_1h_{}_to_{}.csv".format(
    SYMBOL,
    START_DATE,
    (END_DATE or datetime.utcnow().strftime("%Y-%m-%d"))
)

def get_json(url):
    resp = urlopen(url, cafile=certifi.where())
    data = resp.read().decode("utf-8")
    return json.loads(data)

def daterange_chunks(start_dt, end_dt, days_per_chunk):
    """Yield (chunk_start, chunk_end) inclusive date tuples."""
    cur = start_dt
    step = timedelta(days=days_per_chunk)
    one_day = timedelta(days=1)
    while cur <= end_dt:
        chunk_end = min(cur + step - one_day, end_dt)
        yield cur, chunk_end
        cur = chunk_end + one_day

def fetch_chunk(symbol, api_key, dt_from, dt_to, max_retries=3, backoff=1.7):
    """Fetch one window with lightweight retries."""
    params = {
        "symbol": symbol,
        "from": dt_from.strftime("%Y-%m-%d"),
        "to": dt_to.strftime("%Y-%m-%d"),
        "apikey": api_key,
    }
    url = BASE + "?" + urlencode(params)
    last_err = None
    for attempt in range(max_retries):
        try:
            data = get_json(url)
            # Expected: list of bars; sometimes empty list if no data
            if isinstance(data, dict) and data.get("Error Message"):
                raise RuntimeError(data.get("Error Message"))
            return data if isinstance(data, list) else []
        except Exception as e:
            last_err = e
            sleep_s = (backoff ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError("Failed window {}–{}: {}".format(dt_from.date(), dt_to.date(), last_err))

def main():
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = (datetime.utcnow() if END_DATE is None
              else datetime.strptime(END_DATE, "%Y-%m-%d"))

    all_rows = []
    seen_timestamps = set()  # to de-duplicate exact duplicates across windows

    for i, (c_start, c_end) in enumerate(daterange_chunks(start_dt, end_dt, DAYS_PER_CHUNK), start=1):
        try:
            chunk = fetch_chunk(SYMBOL, API_KEY, c_start, c_end, MAX_RETRIES, BACKOFF)
        except Exception as e:
            # Log and continue (or raise if you prefer)
            print("[WARN] {}: {}–{} -> {}".format(i, c_start.date(), c_end.date(), e))
            continue

        # FMP typically returns newest->oldest; normalize and de-dup
        for row in chunk:
            # Expected keys: date, open, high, low, close, volume
            ts = row.get("date")
            if not ts:
                continue
            if ts in seen_timestamps:
                continue
            seen_timestamps.add(ts)
            all_rows.append({
                "timestamp": ts,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            })

        print("Chunk {:4d}: {} – {}  (+{:>5d} rows, total {:>8d})"
              .format(i, c_start.date(), c_end.date(), len(chunk), len(all_rows)))
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Sort ascending by timestamp
    all_rows.sort(key=lambda r: r["timestamp"])

    # Write CSV
    fieldnames = ["timestamp", "open", "high", "low", "close", "volume"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("Saved {:,} rows to {}".format(len(all_rows), OUT_CSV))

if __name__ == "__main__":
    main()
