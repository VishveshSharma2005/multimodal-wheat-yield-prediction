from datetime import timedelta


def build_stage_windows(sowing_date, stage_definitions):
    windows = []
    for stage in stage_definitions:
        start = sowing_date + timedelta(days=stage["start_das"])
        end = sowing_date + timedelta(days=stage["end_das"])
        windows.append(
            {
                "stage": stage["name"],
                "start_date": start,
                "end_date": end,
                "start_das": stage["start_das"],
                "end_das": stage["end_das"],
            }
        )
    return windows
