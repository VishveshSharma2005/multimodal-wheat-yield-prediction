import numpy as np


def linear_slope(x, y):
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    x = x - x.min()
    denom = (x ** 2).sum()
    if denom == 0:
        return np.nan
    return (x * y).sum() / denom
