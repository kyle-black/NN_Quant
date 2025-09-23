import numpy as np
import pandas as pd
from typing import Dict, Any


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

    if margin is not None:
        sorted_p = np.sort(probs, axis=1)
        gap = sorted_p[:, -1] - sorted_p[:, -2]
        ok = (pmax >= pmin) & (gap >= margin)
    else:
        ok = (pmax >= pmin)

    map_idx = np.array([-1, 0, +1])
    sig = map_idx[argmax]
    sig[~ok] = 0
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
) -> pd.DataFrame:
    """
    Turn {-1,0,+1} signals into trades (entries/exits).
    Robust to duplicate timestamps by de-duplicating and aligning on the union index.
    Enters/exits on NEXT bar (1-bar latency).
    """
    close = pd.Series(close).sort_index()
    signals = pd.Series(signals).sort_index()
    if close.index.has_duplicates:
        close = close[~close.index.duplicated(keep="last")]
    if signals.index.has_duplicates:
        signals = signals[~signals.index.duplicated(keep="last")]

    idx = close.index.union(signals.index).sort_values()
    close = close.reindex(idx).astype(float)
    sig = signals.reindex(idx).fillna(0).astype(int)

    exec_sig = sig.shift(1).fillna(0).astype(int)
    change = exec_sig.ne(exec_sig.shift(1).fillna(0))
    change_idx = exec_sig.index[change]

    trades = []
    current_side = 0
    entry_time = None
    entry_price = None
    entry_idx = None
    cost = one_way_cost_bp / 1e4

    for t in change_idx:
        new_side = int(exec_sig.loc[t])
        price = float(close.loc[t])

        if current_side == 0 and new_side != 0:
            # open
            current_side = new_side
            entry_time = t
            entry_price = price
            entry_idx = close.index.get_loc(t)
            continue

        if current_side != 0 and new_side != current_side:
            # close (and possibly flip)
            exit_time = t
            exit_price = price
            bars = close.index.get_loc(exit_time) - entry_idx
            ret_gross = (exit_price / entry_price - 1.0) * current_side
            ret_net = ret_gross - 2 * cost  # entry + exit

            trades.append({
                "side": "LONG" if current_side == 1 else "SHORT",
                "entry_time": entry_time,
                "exit_time": exit_time,
                "bars": bars,
                "duration": (exit_time - entry_time),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret_gross": ret_gross,
                "ret_net": ret_net,
            })

            # flip/open new?
            if new_side != 0:
                current_side = new_side
                entry_time = t
                entry_price = price
                entry_idx = close.index.get_loc(t)
            else:
                current_side = 0
                entry_time = entry_price = entry_idx = None

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades = df_trades.sort_values("entry_time").reset_index(drop=True)
    return df_trades


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
