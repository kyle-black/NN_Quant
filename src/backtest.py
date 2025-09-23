import numpy as np
import pandas as pd
from typing import Dict, Any

def softmax_to_dir(probs: np.ndarray) -> np.ndarray:
    """
    Convert [p(-1), p(0), p(+1)] -> {-1,0,+1} by argmax.
    """
    idx = probs.argmax(axis=1)
    mapping = {0: -1, 1: 0, 2: +1}
    return np.vectorize(mapping.get)(idx)

def pnl_from_signals(close: pd.Series, signals: pd.Series, one_way_cost_bp: float = 0.0) -> Dict[str, Any]:
    """
    Toy PnL: daily ret * signal with basic cost on changes.
    Costs modeled in basis points per trade side.
    """
    close = close.reindex(signals.index).astype(float)
    ret = close.pct_change().fillna(0.0)

    # apply costs when signal changes
    sig = signals.fillna(0).astype(int)
    changes = (sig != sig.shift()).astype(int)
    costs = changes * (one_way_cost_bp / 1e4)  # convert bp to fraction

    strat_ret = sig.shift().fillna(0).astype(float) * ret - costs
    equity = (1.0 + strat_ret).cumprod()
    dd = equity / equity.cummax() - 1.0

    return {
        "n_trades": int(changes.sum()),
        "avg_ret": float(strat_ret.mean()),
        "cum_ret": float((equity.iloc[-1] - 1.0)),
        "sharpe": float(np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-12)),
        "max_dd": float(dd.min()),
    }
