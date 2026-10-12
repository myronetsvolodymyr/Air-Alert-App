import numpy as np
import pandas as pd

from airalert.pipeline import build_features, build_grid


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=400, freq="h", tz="UTC")
    raions = ["A", "B", "C", "D"]
    n = 300
    start = hours[0] + pd.to_timedelta(rng.uniform(0, 390, n), unit="h")
    end = start + pd.to_timedelta(rng.uniform(0.1, 12, n), unit="h")
    intervals = pd.DataFrame({"raion_id": rng.choice(raions, n), "start": start, "end": end})
    spatial = {
        "ring1": np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], np.float32),
        "ring2": np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], np.float32),
        "dist": np.array([[0, 100, 250, 400], [100, 0, 150, 300], [250, 150, 0, 150], [400, 300, 150, 0]], float),
        "same_oblast": np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], np.float32),
    }
    return hours, raions, intervals, spatial


def test_features_at_t_do_not_depend_on_the_future():
    # AC-5.4: прибираємо все, що сталося після T, — ознаки на T і раніше не змінюються.
    # Тривога, що триває в T, лишається «без відбою»: її кінець на T ще невідомий (AC-5.2).
    hours, raions, intervals, spatial = _synthetic()
    full = build_features(*build_grid(intervals, hours, raions), hours, spatial)

    for t in (150, 260, 333):
        cutoff = hours[t]
        known = intervals[intervals["start"] <= cutoff].copy()
        known.loc[known["end"] > cutoff, "end"] = hours[-1] + pd.Timedelta(days=30)
        cut = build_features(*build_grid(known, hours, raions), hours, spatial)
        for name in full:
            np.testing.assert_array_equal(full[name][: t + 1], cut[name][: t + 1], err_msg=f"{name} at T={cutoff}")
