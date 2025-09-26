import numpy as np
import pandas as pd

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-only features on 1h bars + a D1 ATR usable for day-scale barriers.
    Requires columns: open, high, low, close (case-insensitive).

    New features added:
      - RSI(14): rsi_14
      - Williams %R(14): willr_14
      - Bollinger Bands(20, 2σ): bb_mid_20, bb_up_20_2, bb_dn_20_2, bb_pctb_20_2, bb_bw_20_2
      - Normalized ATR(14): natr_14  = 100 * atr_14 / close
      - PPO(12,26,9): ppo, ppo_signal, ppo_hist   (all in %)
      - CCI(20): cci_20
      - (retained) MACD(12,26,9): macd, macd_signal, macd_hist
      - (retained) ATR(14) 1h: atr_14
      - (retained) Daily ATR(14): atr_d1_14 (forward-filled within day)
      - Plus returns/momentum/rolling stats you already had
    """
    out = df.copy()

    # normalize column names but keep original keys
    colmap = {c.lower(): c for c in out.columns}
    for k in ("open", "high", "low", "close"):
        if k not in colmap:
            raise KeyError(f"Missing column '{k}' in data")
    O, H, L, C = [colmap[x] for x in ("open", "high", "low", "close")]

    # ---------- Basic returns & rolling stats ----------
    out["ret_1"] = np.log(out[C]).diff()
    for w in (5, 10, 20, 50, 100):
        out[f"roll_mean_{w}"] = out["ret_1"].rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"]  = out["ret_1"].rolling(w, min_periods=w).std()

    # Momentum windows
    for k in (2, 5, 10, 24):
        out[f"mom_{k}"] = out[C].pct_change(k)

    # ---------- RSI(14) ----------
    delta = out[C].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_dn = down.rolling(14, min_periods=14).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ---------- MACD(12,26,9) ----------
    ema12 = _ema(out[C], 12)
    ema26 = _ema(out[C], 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # ---------- ATR(14) on 1h ----------
    prev_close = out[C].shift(1)
    tr1 = (out[H] - out[L]).abs()
    tr2 = (out[H] - prev_close).abs()
    tr3 = (out[L] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()

    # ---------- Daily ATR(14) (resample to D1 and ffill back) ----------
    d1 = out[[O, H, L, C]].resample("1D").agg({O: "first", H: "max", L: "min", C: "last"}).dropna()
    prev_close_d1 = d1[C].shift(1)
    tr_d1 = pd.concat([
        (d1[H] - d1[L]).abs(),
        (d1[H] - prev_close_d1).abs(),
        (d1[L] - prev_close_d1).abs()
    ], axis=1).max(axis=1)
    atr_d1_14 = tr_d1.rolling(14, min_periods=14).mean()
    out["atr_d1_14"] = atr_d1_14.reindex(out.index, method="ffill")

    # ---------- Williams %R(14) ----------
    hh_14 = out[H].rolling(14, min_periods=14).max()
    ll_14 = out[L].rolling(14, min_periods=14).min()
    out["willr_14"] = -100 * (hh_14 - out[C]) / (hh_14 - ll_14).replace(0, np.nan)

    # ---------- Bollinger Bands (20, 2σ) ----------
    bb_mid = out[C].rolling(20, min_periods=20).mean()
    bb_std = out[C].rolling(20, min_periods=20).std()
    out["bb_mid_20"] = bb_mid
    out["bb_up_20_2"] = bb_mid + 2.0 * bb_std
    out["bb_dn_20_2"] = bb_mid - 2.0 * bb_std
    # %B = (price - lower) / (upper - lower)
    denom = (out["bb_up_20_2"] - out["bb_dn_20_2"]).replace(0, np.nan)
    out["bb_pctb_20_2"] = (out[C] - out["bb_dn_20_2"]) / denom
    # Bandwidth = (upper - lower) / mid
    out["bb_bw_20_2"] = denom / bb_mid.replace(0, np.nan)

    # ---------- Normalized ATR (NATR) ----------
    out["natr_14"] = 100 * (out["atr_14"] / out[C]).replace(0, np.nan)

    # ---------- Percentage Price Oscillator (PPO 12,26,9) ----------
    ema_fast = _ema(out[C], 12)
    ema_slow = _ema(out[C], 26)
    ppo = 100 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    ppo_signal = _ema(ppo, 9)
    out["ppo"] = ppo
    out["ppo_signal"] = ppo_signal
    out["ppo_hist"] = ppo - ppo_signal

    # ---------- Commodity Channel Index (CCI 20) ----------
    tp = (out[H] + out[L] + out[C]) / 3.0
    sma_tp_20 = tp.rolling(20, min_periods=20).mean()
    mean_dev = (tp - sma_tp_20).abs().rolling(20, min_periods=20).mean()
    out["cci_20"] = (tp - sma_tp_20) / (0.015 * mean_dev.replace(0, np.nan))

    # Final cleanup: drop warmup rows created by rolling windows
    out = out.dropna().copy()
    return out
