# backtest.py
# Utilities for turning model outputs into signals, parsing trades, and computing PnL.
# Includes:
# - softmax_to_dir_threshold          (existing, kept)
# - softmax_to_dir_longshort_gap      (NEW)   -> ignore neutral; trade only if Long vs Short gap big enough
# - confidence_scaled_mult            (NEW)   -> scale ATR barrier by L–S confidence per entry
# - pnl_from_signals                  (kept)  -> portfolio metrics with 1-bar latency
# - pnl_timeseries                    (kept)  -> daily equity series (flat on non-trading days)
# - trades_from_signals               (UPDATED) supports per-entry barrier multipliers
# - summarize_trades                  (kept)

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# -----------------------------
# Signal mappers
# -----------------------------
def softmax_to_dir_threshold(
    probs: np.ndarray,
    pmin: float = 0.60,
    margin: Optional[float] = 0.05
) -> np.ndarray:
    """
    Convert 3-class probs [p_short, p_neutral, p_long] -> {-1,0,+1}.
    - act only if top prob >= pmin
    - if margin is not None: require (top - second) >= margin
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be (N,3) for [short, neutral, long].")

    top_idx = probs.argmax(axis=1)
    top_val = probs.max(axis=1)
    # abstain if top prob < pmin
    out = np.where(top_val >= pmin, 0, 0)

    if margin is not None:
        # compute second best
        sorted_idx = np.argsort(probs, axis=1)  # ascending
        second_val = probs[np.arange(len(probs)), sorted_idx[:, -2]]
        ok = (top_val - second_val) >= margin
    else:
        ok = top_val >= pmin

    # map class index -> dir
    # 0: short -> -1, 1: neutral -> 0, 2: long -> +1
    cls_to_dir = np.array([-1, 0, +1])
    dir_all = cls_to_dir[top_idx]
    out = np.where(ok, dir_all, 0)
    return out.astype(int)


def softmax_to_dir_longshort_gap(
    probs: np.ndarray,
    pmin_ls: float = 0.60,
    gap: float = 0.05
) -> np.ndarray:
    """
    NEW: Long–Short gap rule that ignores neutral.
    probs = [p_short, p_neutral, p_long]
    Rules:
      - best = max(p_long, p_short)
      - if best < pmin_ls -> abstain (0)
      - if (p_long - p_short) > gap -> +1
      - if (p_short - p_long) > gap -> -1
      - else -> 0
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be (N,3) for [short, neutral, long].")

    pS = probs[:, 0]
    pL = probs[:, 2]
    best = np.maximum(pS, pL)

    out = np.zeros(len(probs), dtype=int)
    long_mask  = ((pL - pS) > gap) & (best >= pmin_ls)
    short_mask = ((pS - pL) > gap) & (best >= pmin_ls)

    out[long_mask]  = +1
    out[short_mask] = -1
    return out


def confidence_scaled_mult(
    p_long: float,
    p_short: float,
    base_k: float,
    gap: float,
    k_min: float = 0.5,
    k_max: float = 3.0,
) -> float:
    """
    NEW: Scale ATR barrier multiplier by Long–Short confidence.
    conf = max(|pL - pS|, gap); k = base_k * (conf / gap), clipped to [k_min, k_max].
    """
    conf = abs(float(p_long) - float(p_short))
    conf = max(conf, float(gap))
    k = base_k * (conf / gap)
    return float(np.clip(k, k_min, k_max))


# -----------------------------
# PnL helpers
# -----------------------------
def _safe_align(close: pd.Series, signals: pd.Series) -> tuple[pd.Series, pd.Series]:
    # 1-bar latency: use signal at t to trade at t+1 close
    sig = signals.reindex(close.index).fillna(0).astype(int)
    sig = sig.shift(1).fillna(0).astype(int)  # latency
    return close, sig


def pnl_from_signals(
    close: pd.Series,
    signals: pd.Series,
    one_way_cost_bp: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute simple equity curve from directional signals on close-to-close returns.
    Applies 1-bar latency and fixed bps costs on position changes (entry + exit side).
    """
    close = close.astype(float).copy()
    close, sig = _safe_align(close, signals)

    ret = close.pct_change().fillna(0.0)
    pos_change = sig.diff().fillna(0).abs()
    cost = (one_way_cost_bp / 1e4) * pos_change  # charged on changes
    strat_ret = sig.shift(0).fillna(0) * ret - cost

    equity = (1.0 + strat_ret).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0

    tot_ret = float(equity.iloc[-1] - 1.0)
    vol = float(strat_ret.std() * np.sqrt(252))
    sharpe = float((strat_ret.mean() * 252) / (strat_ret.std() + 1e-12))
    maxdd = float(drawdown.min())
    cagr = float((equity.iloc[-1]) ** (252 / max(1, len(equity))) - 1.0)

    return {
        "total_return": tot_ret,
        "sharpe": sharpe,
        "vol": vol,
        "max_drawdown": maxdd,
        "cagr": cagr,
        "n_days": int(len(equity)),
    }


def pnl_timeseries(
    close: pd.Series,
    signals: pd.Series,
    one_way_cost_bp: float = 0.0,
    resample: str = "D",
    start_equity: float = 1.0,
) -> pd.DataFrame:
    """
    Daily (or chosen frequency) equity series, flat on non-trading days.
    """
    close = close.astype(float).copy()
    close, sig = _safe_align(close, signals)
    ret = close.pct_change().fillna(0.0)
    pos_change = sig.diff().fillna(0).abs()
    cost = (one_way_cost_bp / 1e4) * pos_change
    strat_ret = sig.shift(0).fillna(0) * ret - cost
    eq = (start_equity * (1.0 + strat_ret).cumprod()).rename("equity")

    df = pd.DataFrame({"equity": eq, "ret_net": strat_ret})
    if resample:
        df = df.resample(resample).last().ffill()
        df["ret_net"] = df["equity"].pct_change().fillna(0.0)
    roll_max = df["equity"].cummax()
    df["drawdown"] = df["equity"] / roll_max - 1.0
    return df


# -----------------------------
# Trades parsing
# -----------------------------
def trades_from_signals(
    close: pd.Series,
    signals: pd.Series,
    one_way_cost_bp: float = 0.0,
    max_hold_bars: Optional[int] = None,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    entry_barrier_mult: Optional[float] = None,     # base k * daily_atr at entry
    daily_atr: Optional[pd.Series] = None,          # typically atr_d1_14 on 1h index (ffilled)
    per_entry_mult: Optional[pd.Series] = None,     # NEW: Series of k_at_entry aligned to signal timestamps
) -> pd.DataFrame:
    """
    Parse signal stream into discrete trades with optional:
      - time exit (max_hold_bars)
      - barrier exits sized by (k * daily_atr at entry). If per_entry_mult given, it overrides base.
    Uses 1-bar latency: entries occur at the next bar after a non-zero signal appears.

    Returns DataFrame with columns:
      ['side','entry_time','entry_price','exit_time','exit_price',
       'gross_ret','net_ret','hold_bars','reason','k_mult']
    """
    idx = close.index
    sig = signals.reindex(idx).fillna(0).astype(int)
    # 1-bar latency
    sig_lag = sig.shift(1).fillna(0).astype(int)

    high = high.reindex(idx) if high is not None else None
    low  = low.reindex(idx)  if low is not None  else None
    atr  = daily_atr.reindex(idx) if daily_atr is not None else None

    # Prepare iteration
    trades = []
    in_pos = 0
    entry_i = None
    entry_px = None
    side = 0
    k_used = None
    entry_barrier = None  # (upper, lower)

    def _calc_k_at_entry(tstamp) -> Optional[float]:
        if per_entry_mult is not None and tstamp in per_entry_mult.index:
            val = per_entry_mult.loc[tstamp]
            try:
                return float(val)
            except Exception:
                return None
        return float(entry_barrier_mult) if entry_barrier_mult is not None else None

    def _make_barriers(px: float, side_: int, k: Optional[float]) -> Optional[tuple[float, float]]:
        if k is None or atr is None:
            return None
        a = atr.loc[i] if i in atr.index else np.nan
        if pd.isna(a):
            return None
        if side_ > 0:
            # long: TP = px + k*a, SL = px - k*a
            return (px + k * a, px - k * a)
        elif side_ < 0:
            # short: TP = px - k*a, SL = px + k*a
            return (px - k * a, px + k * a)
        return None

    for i in idx:
        s = sig_lag.loc[i]  # effective position for bar i
        px = float(close.loc[i])

        # open new position?
        if in_pos == 0 and s != 0:
            in_pos = s
            side = s
            entry_i = i
            entry_px = px
            k_used = _calc_k_at_entry(entry_i)
            entry_barrier = _make_barriers(entry_px, side, k_used)
            hold = 0
            continue

        # if in position, check exits
        if in_pos != 0 and entry_i is not None:
            hold += 1

            # Exit condition 1: opposite or flat signal (position flip)
            if s == 0 or np.sign(s) != np.sign(in_pos):
                trades.append({
                    "side": in_pos,
                    "entry_time": entry_i,
                    "entry_price": entry_px,
                    "exit_time": i,
                    "exit_price": px,
                    "gross_ret": (px / entry_px - 1.0) * (1 if in_pos > 0 else -1),
                    "net_ret": (px / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                              - 2 * (one_way_cost_bp / 1e4),  # entry+exit
                    "hold_bars": hold,
                    "reason": "signal_flip",
                    "k_mult": k_used,
                })
                in_pos = 0
                entry_i = None
                continue

            # Exit condition 2: barrier touch (need high/low & atr)
            if entry_barrier is not None and high is not None and low is not None:
                hi = float(high.loc[i])
                lo = float(low.loc[i])
                up, dn = entry_barrier  # (take-profit, stop)
                hit_tp = (hi >= up) if in_pos > 0 else (lo <= up)  # up=profit target
                hit_sl = (lo <= dn) if in_pos > 0 else (hi >= dn)

                if hit_tp or hit_sl:
                    exit_px = up if hit_tp else dn
                    trades.append({
                        "side": in_pos,
                        "entry_time": entry_i,
                        "entry_price": entry_px,
                        "exit_time": i,
                        "exit_price": exit_px,
                        "gross_ret": (exit_px / entry_px - 1.0) * (1 if in_pos > 0 else -1),
                        "net_ret": (exit_px / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                                  - 2 * (one_way_cost_bp / 1e4),
                        "hold_bars": hold,
                        "reason": "barrier",
                        "k_mult": k_used,
                    })
                    in_pos = 0
                    entry_i = None
                    continue

            # Exit condition 3: time stop
            if (max_hold_bars is not None) and (hold >= max_hold_bars):
                trades.append({
                    "side": in_pos,
                    "entry_time": entry_i,
                    "entry_price": entry_px,
                    "exit_time": i,
                    "exit_price": px,
                    "gross_ret": (px / entry_px - 1.0) * (1 if in_pos > 0 else -1),
                    "net_ret": (px / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                              - 2 * (one_way_cost_bp / 1e4),
                    "hold_bars": hold,
                    "reason": "time",
                    "k_mult": k_used,
                })
                in_pos = 0
                entry_i = None
                continue

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    return trades_df


def summarize_trades(trades_df: pd.DataFrame) -> Dict[str, Any]:
    if trades_df is None or trades_df.empty:
        return {"n": 0}
    longs = trades_df[trades_df["side"] > 0]
    shorts = trades_df[trades_df["side"] < 0]
    return {
        "n": int(len(trades_df)),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "avg_win": float(trades_df.loc[trades_df["net_ret"] > 0, "net_ret"].mean() or 0.0),
        "avg_loss": float(trades_df.loc[trades_df["net_ret"] <= 0, "net_ret"].mean() or 0.0),
        "avg_hold_bars": float(trades_df["hold_bars"].mean() or 0.0),
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
    }
