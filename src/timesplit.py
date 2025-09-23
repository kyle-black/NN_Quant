import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def time_series_split(n: int, n_splits: int = 5, test_size: float = 0.2) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window split (no shuffling).
    Each split grows the train set; test is a fixed tail proportion.
    """
    idx = np.arange(n)
    test_n = int(max(1, n * test_size))
    for i in range(n_splits):
        end_train = int((n - test_n) * ((i + 1) / n_splits))
        if end_train <= 1:
            continue
        train_idx = idx[:end_train]
        test_idx  = idx[end_train:end_train+test_n]
        if len(test_idx) == 0:
            break
        yield train_idx, test_idx
