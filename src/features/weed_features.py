def compute_crop_only_ndvi(ndvi_raw, weed_cover_pct):
    if ndvi_raw is None:
        return None
    if weed_cover_pct is None:
        return None
    try:
        weed_frac = float(weed_cover_pct) / 100.0
    except (TypeError, ValueError):
        return None
    return ndvi_raw * (1.0 - weed_frac)
