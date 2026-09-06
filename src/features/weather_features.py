import numpy as np


def compute_gdd(tmin_c, tmax_c, t_base=0.0):
    tmean = (tmin_c + tmax_c) / 2.0
    gdd = np.maximum(0.0, tmean - t_base)
    return gdd
