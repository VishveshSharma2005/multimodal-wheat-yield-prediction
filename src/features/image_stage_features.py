def map_stage_label(label, mapping):
    if label is None:
        return None
    return mapping.get(label, None)
