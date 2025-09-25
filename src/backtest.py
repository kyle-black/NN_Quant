import numpy as np
import pandas as pd
from typing import Dict, Any

# ... keep your existing softmax_to_dir_threshold and pnl_from_signals ...
def softmax_to_dir(probs: np.ndarray) -> np.ndarray:
    """
    Convert [p(-1), p(0), p(+1)] -> {-1,0,+1} by argmax.
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be (N, 3) with classes [-1, 0, +1].")
    idx = probs.argmax(axis=1)
    mapping = {0: -1, 1: 0, 2: +1}
    return np.vectorize(mapping.get)(idx)


def softmax_to_dir_threshold(
    probs: np.ndarray,
    pmin: float = 0.60,
    margin: float | None = 0.05,
) -> np.ndarray:
    """
    Thresholded conversion of [p(-1), p(0), p(+1)] -> {-1,0,+1}.
      - Requires max(prob) >= pmin
      - If margin is provided, also requires (p_top - p_second) >= margin
      - Otherwise abstains (0).
    """
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be (N, 3) with classes [-1, 0, +1].")

    argmax = probs.argmax(axis=1)
    pmax = probs.max(axis=1)
    print(f'probs {probs}')
    if margin is not None:
        sorted_p = np.sort(probs, axis=1)
        print(sorted_p)
        gap = sorted_p[:, -1] - sorted_p[:, -2]
        ok = (pmax >= pmin) & (gap >= margin)
    else:
        ok = (pmax >= pmin)

    map_idx = np.array([-1, 0, +1])
    sig = map_idx[argmax]
    sig[~ok] = 0
    print(sig)
    return sig


def pnl_from_signals(
    close: pd.Series,
    signals: pd.Series,
    one_way_cost_bp: float = 0.5
) -> Dict[str, Any]:
    """
    Simple PnL: ret * lagged_signal with transaction costs on changes.
    Costs are basis-points per side (one-way).
    """
    # align on union & de-duplicate to be robust
    close = pd.Series(close).sort_index()
    signals = pd.Series(signals).sort_index()
    if close.index.has_duplicates:
        close = close[~close.index.duplicated(keep="last")]
    if signals.index.has_duplicates:
        signals = signals[~signals.index.duplicated(keep="last")]
    idx = close.index.union(signals.index).sort_values()
    close = close.reindex(idx).astype(float)
    sig = signals.reindex(idx).fillna(0).astype(int)

    ret = close.pct_change().fillna(0.0)
    changes = (sig != sig.shift()).astype(int)
    cost = one_way_cost_bp / 1e4

    # 1-bar latency execution to avoid look-ahead
    strat_ret = sig.shift().fillna(0).astype(float) * ret - changes * cost
    equity = (1.0 + strat_ret).cumprod()
    dd = equity / equity.cummax() - 1.0

    return {
        "n_trades": int(changes.sum()),
        "avg_ret": float(strat_ret.mean()),
        "cum_ret": float(equity.iloc[-1] - 1.0),
        "sharpe": float(np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)),
        "max_dd": float(dd.min()),
    }


def trades_from_signals(
    close: pd.Series,
    signals: pd.Series,
    one_way_cost_bp: float = 0.5,
    max_hold_bars: int | None = None,        # NEW: time exit in hours (1 bar = 1h)
    high: pd.Series | None = None,            # for barrier exits
    low: pd.Series | None = None,
    entry_barrier_mult: float | None = None,  # if set with daily_atr, use triple barrier
    daily_atr: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Build trades with 1-bar latency.
    - If max_hold_hours is set: force time exit after N hours.
    - If entry_barrier_mult & daily_atr provided: early exit on first-touch
      of entry_price +/- k * daily_atr_at_entry.
    """
    close = pd.Series(close).sort_index().astype(float)
    sig = pd.Series(signals).sort_index().fillna(0).astype(int)

    # align indices
    idx = close.index.union(sig.index).sort_values()
    close = close.reindex(idx)
    sig = sig.reindex(idx).fillna(0).astype(int)

    if high is not None and low is not None:
        high = pd.Series(high).reindex(idx).astype(float)
        low  = pd.Series(low ).reindex(idx).astype(float)
    if daily_atr is not None:
        daily_atr = pd.Series(daily_atr).reindex(idx).astype(float)

    # 1-bar latency
    exec_sig = sig.shift(1).fillna(0).astype(int)

    trades = []
    cur = 0
    entry_i = None
    entry_t = None
    entry_px = None
    cost = one_way_cost_bp / 1e4

    for i, t in enumerate(idx):
        s = int(exec_sig.iloc[i])
        px = float(close.iloc[i])

        # open?
        if cur == 0 and s != 0:
            cur = s
            entry_i = i
            entry_t = t
            entry_px = px
            # set static barriers at entry if enabled
            up_bar = dn_bar = None
            if entry_barrier_mult is not None and daily_atr is not None:
                atr0 = float(daily_atr.iloc[i])
                if np.isfinite(atr0):
                    up_bar = entry_px + entry_barrier_mult * atr0
                    dn_bar = entry_px - entry_barrier_mult * atr0
            continue

        # manage open position
        if cur != 0:
            flip_exit = (s != cur)
            time_exit = (max_hold_bars is not None and entry_i is not None and (i - entry_i) >= max_hold_bars)

            barrier_exit = False
            if entry_barrier_mult is not None and daily_atr is not None and high is not None and low is not None:
                # check first-touch on this bar
                if cur == 1 and np.isfinite(up_bar) and high.iloc[i] >= up_bar:
                    barrier_exit = True
                elif cur == -1 and np.isfinite(dn_bar) and low.iloc[i] <= dn_bar:
                    barrier_exit = True

            if flip_exit or time_exit or barrier_exit:
                exit_t = t
                exit_px = px
                bars = i - entry_i
                ret_gross = (exit_px / entry_px - 1.0) * cur
                ret_net = ret_gross - 2 * cost

                trades.append({
                    "side": "LONG" if cur == 1 else "SHORT",
                    "entry_time": entry_t, "exit_time": exit_t,
                    "bars": bars, "duration": (exit_t - entry_t),
                    "entry_price": entry_px, "exit_price": exit_px,
                    "ret_gross": ret_gross, "ret_net": ret_net,
                    "exit_reason": ("flip" if flip_exit else ("time" if time_exit else "barrier")),
                })

                # flip open immediately if signal changed
                if s != 0 and not time_exit:
                    cur = s
                    entry_i = i
                    entry_t = t
                    entry_px = px
                    up_bar = dn_bar = None
                    if entry_barrier_mult is not None and daily_atr is not None:
                        atr0 = float(daily_atr.iloc[i])
                        if np.isfinite(atr0):
                            up_bar = entry_px + entry_barrier_mult * atr0
                            dn_bar = entry_px - entry_barrier_mult * atr0
                else:
                    cur = 0
                    entry_i = entry_t = entry_px = None

    return pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
def summarize_trades(trades_df: pd.DataFrame) -> dict:
    """
    Summary stats overall and by side.
    """
    def _summary(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        wins = (df["ret_net"] > 0).sum()
        return {
            "n": int(len(df)),
            "win_rate": float(wins / len(df)),
            "avg_ret_net": float(df["ret_net"].mean()),
            "med_ret_net": float(df["ret_net"].median()),
            "avg_bars": float(df["bars"].mean()),
            "med_bars": float(df["bars"].median()),
            "avg_duration_min": float(df["duration"].mean().total_seconds() / 60.0),
        }

    return {
        "overall": _summary(trades_df),
        "long": _summary(trades_df[trades_df["side"] == "LONG"]),
        "short": _summary(trades_df[trades_df["side"] == "SHORT"]),
    }