import pandas as pd
import data_prep as dp

def _toy():
    return pd.DataFrame({
        "cycle": [2, 2, 2, 4, 4, 4],
        "time": [0, 10, 20, 0, 10, 20],
        "temperature": [30, 41, 42, 30, 31, 32],
        "thermal_event": [0, 1, 1, 0, 0, 0],
    })


def test_cycle_table_counts():
    t = dp.battery_cycle_table(_toy())
    assert t.loc[t.cycle == 2, "events"].iloc[0] == 2
    assert t.loc[t.cycle == 4, "events"].iloc[0] == 0
    assert t.loc[t.cycle == 2, "first_event_s"].iloc[0] == 10


def test_threshold_recompute():
    t = dp.battery_cycle_table(_toy(), threshold=41.5)
    assert t.loc[t.cycle == 2, "events"].iloc[0] == 1


def test_detect_format():
    assert dp.is_battery_file(_toy())
