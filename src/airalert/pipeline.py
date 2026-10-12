"""MVP: тривоги → сітка «район × година» → ознаки станом на T → модель → метрики.

Кроки 3–5 плану MVP (JS-1, JS-2, JS-5, JS-6). Запуск: python -m airalert.pipeline
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from airalert import config
from airalert.reference import _haversine_km

KYIV = "Europe/Kyiv"
HISTORY_H = 168  # найдовше вікно ознак — тиждень
WAVE_RADIUS_KM = 200  # радіус R з AC-6.6; у MVP не підбирали
TARGET_SHARE = (0.01, 0.95)  # AC-2.4
FRONTLINE_QUANTILE = 0.75  # AC-6.7: верхня чверть районів за часткою часу під тривогою
NO_ALERT_KM = 1500.0  # «відстань до найближчої тривоги», коли тривог немає ніде


# ---------- Крок 3: очистка й сітка ----------


def load_intervals() -> pd.DataFrame:
    """Інтервали тривог на рівні районів: raion_id, start, end (AC-2.2, AC-2.6, D-14)."""
    raw = pd.read_csv(config.OFFICIAL_ALERTS_RAW).drop_duplicates()
    raw["start"] = pd.to_datetime(raw["started_at"], utc=True)
    raw["end"] = pd.to_datetime(raw["finished_at"], utc=True)
    raw = raw[(raw["end"] > raw["start"]) & raw["level"].isin(["oblast", "raion"])]

    aliases = pd.read_csv(config.REFERENCE / "name_aliases.csv")
    raions = pd.read_csv(config.REFERENCE / "raions.csv")
    oblast_ids = dict(zip(aliases["oblast_src"], aliases["oblast_id"]))
    raion_ids = {
        (o, r): i
        for o, r, i in aliases.dropna(subset=["raion_src"])[["oblast_src", "raion_src", "raion_id"]].itertuples(index=False)
    }

    by_raion = raw[raw["level"] == "raion"].copy()
    by_raion["raion_id"] = [raion_ids[(o, r)] for o, r in zip(by_raion["oblast"], by_raion["raion"])]
    # Обласна тривога поширюється на всі райони області (AC-2.6)
    by_oblast = raw[raw["level"] == "oblast"].copy()
    by_oblast["oblast_id"] = by_oblast["oblast"].map(oblast_ids)
    by_oblast = by_oblast.merge(raions[["raion_id", "oblast_id"]], on="oblast_id")

    out = pd.concat([by_raion, by_oblast])[["raion_id", "start", "end"]]
    return out.sort_values(["raion_id", "start"]).reset_index(drop=True)


def build_grid(intervals: pd.DataFrame, hours: pd.DatetimeIndex, raion_ids: list[str]):
    """Матриці H×R: тривога активна на початку години; кількість нових тривог за попередню годину.

    Перекриті інтервали району об'єднуємо (AC-2.2). Стан на T залежить лише від подій до T:
    для тривоги, що триває, важливо тільки те, що відбою ще не було (AC-5.2).
    """
    h_ns = hours.as_unit("ns").asi8
    active = np.zeros((len(hours), len(raion_ids)), dtype=np.float32)
    starts = np.zeros_like(active)
    column = {r: j for j, r in enumerate(raion_ids)}
    for raion, group in intervals.groupby("raion_id"):
        if raion not in column:
            continue
        j = column[raion]
        s = group["start"].to_numpy("datetime64[ns]").astype(np.int64)
        e = group["end"].to_numpy("datetime64[ns]").astype(np.int64)
        order = np.argsort(s, kind="stable")
        s, e = s[order], np.maximum.accumulate(e[order])
        new_block = np.r_[True, s[1:] > e[:-1]]
        block_start = s[new_block]
        block_end = e[np.r_[new_block[1:], True]]

        k = np.searchsorted(block_start, h_ns, side="right") - 1
        active[:, j] = (k >= 0) & (block_end[np.maximum(k, 0)] > h_ns)
        t = np.searchsorted(h_ns, block_start, side="right")  # старт ∈ [T−1 год, T)
        np.add.at(starts[:, j], t[t < len(hours)], 1)
    return active, starts


def _rolling(x: np.ndarray, k: int) -> np.ndarray:
    """Сума за останні k годин, включно з поточною — лише минуле."""
    c = np.vstack([np.zeros((1, x.shape[1]), x.dtype), np.cumsum(x, axis=0)])
    i = np.arange(1, x.shape[0] + 1)
    return c[i] - c[np.maximum(i - k, 0)]


# ---------- Крок 4: ознаки станом на T ----------


def spatial_matrices(raion_ids: list[str]) -> dict[str, np.ndarray]:
    """Сусідство, відстані й належність до області з довідників (AC-6.6)."""
    pos = {r: j for j, r in enumerate(raion_ids)}
    raions = pd.read_csv(config.REFERENCE / "raions.csv").set_index("raion_id").loc[raion_ids]
    neighbors = pd.read_csv(config.REFERENCE / "raion_neighbors.csv")
    neighbors = neighbors[neighbors["raion_id"].isin(pos) & neighbors["neighbor_id"].isin(pos)]

    n = len(raion_ids)
    rings = {}
    for ring in (1, 2):
        m = np.zeros((n, n), np.float32)
        sub = neighbors[neighbors["ring"] == ring]
        m[sub["raion_id"].map(pos), sub["neighbor_id"].map(pos)] = 1
        rings[ring] = m
    lat, lon = raions["lat"].to_numpy(), raions["lon"].to_numpy()
    dist = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    same_oblast = (raions["oblast_id"].to_numpy()[:, None] == raions["oblast_id"].to_numpy()[None, :]).astype(np.float32)
    return {"ring1": rings[1], "ring2": rings[2], "dist": dist, "same_oblast": same_oblast}


def build_features(active, starts, hours, spatial) -> dict[str, np.ndarray]:
    """Ознаки H×R. Рядок T використовує лише рядки ≤ T (AC-5.1)."""
    n_hours, n_raions = active.shape
    idx = np.arange(n_hours)[:, None]
    started = (starts > 0).astype(np.float32)
    f = {"active": active, "starts_1h": starts}

    # Історія самого району
    for k in (6, 24, HISTORY_H):
        f[f"active_{k}h"] = _rolling(active, k)
    f["starts_24h"] = _rolling(starts, 24)
    last_start = np.maximum.accumulate(np.where(starts > 0, idx, -HISTORY_H), axis=0)
    f["hours_since_start"] = np.minimum(idx - last_start, HISTORY_H).astype(np.float32)
    last_quiet = np.maximum.accumulate(np.where(active == 0, idx, -HISTORY_H), axis=0)
    f["alert_age_h"] = np.where(active > 0, np.minimum(idx - last_quiet, HISTORY_H), 0).astype(np.float32)

    # Сусіди й поширення загрози (AC-6.6)
    for ring in (1, 2):
        m = spatial[f"ring{ring}"]
        f[f"nb{ring}_active"] = active @ (m / np.maximum(m.sum(1, keepdims=True), 1)).T
        f[f"nb{ring}_starts_1h"] = started @ m.T
    same = spatial["same_oblast"]
    f["oblast_active"] = active @ (same / same.sum(1, keepdims=True)).T
    near = ((spatial["dist"] <= WAVE_RADIUS_KM) & (spatial["dist"] > 0)).astype(np.float32)
    f["wave_starts_1h"] = started @ near.T
    f["wave_starts_3h"] = _rolling(f["wave_starts_1h"], 3)
    dist = spatial["dist"] + np.diag(np.full(n_raions, np.inf))
    nearest = np.empty_like(active)
    for a in range(0, n_hours, 512):
        block = active[a : a + 512]
        nearest[a : a + 512] = np.where(block[:, None, :] > 0, dist[None], np.inf).min(axis=2)
    f["nearest_alert_km"] = np.minimum(nearest, NO_ALERT_KM)

    # Уся країна
    f["national_active"] = np.repeat(active.mean(1, keepdims=True), n_raions, axis=1)
    national_starts = np.repeat(started.sum(1, keepdims=True), n_raions, axis=1)
    f["national_starts_1h"] = national_starts
    f["national_starts_3h"] = _rolling(national_starts, 3)

    # Календар за київським часом
    local = hours.tz_convert(KYIV)
    f["hour"] = np.repeat(local.hour.to_numpy()[:, None], n_raions, axis=1).astype(np.float32)
    f["weekday"] = np.repeat(local.dayofweek.to_numpy()[:, None], n_raions, axis=1).astype(np.float32)
    return f


# ---------- Крок 5: модель і метрики ----------


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    both = 0 < y.mean() < 1
    return {
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p) if both else np.nan,
        "pr_auc": average_precision_score(y, p) if both else np.nan,
        "n": len(y),
        "positive_rate": y.mean(),
    }


def _in_outage(times: pd.DatetimeIndex) -> np.ndarray:
    mask = np.zeros(len(times), bool)
    for start, end, _ in config.OUTAGES:
        mask |= (times >= start) & (times < end)
    return mask


def run() -> pd.DataFrame:
    config.ensure_dirs()
    intervals = load_intervals()
    data_end = intervals["end"].max().floor("h")
    horizon_max = max(config.HORIZONS_H)
    grid_start = config.TRAIN_START - timedelta(hours=HISTORY_H)
    grid_end = min(config.TEST_END + timedelta(hours=horizon_max), data_end)
    hours = pd.date_range(grid_start, grid_end, freq="h", tz="UTC")

    # Райони з даними; Крим, Севастополь і Луганщина мають постійну тривогу, якої в датасеті немає
    in_period = intervals[(intervals["end"] > grid_start) & (intervals["start"] < grid_end)]
    raion_ids = sorted(in_period["raion_id"].unique())
    active, starts = build_grid(intervals, hours, raion_ids)
    features = build_features(active, starts, hours, spatial_matrices(raion_ids))
    names = list(features)

    t0 = hours.get_loc(config.TRAIN_START)
    t1 = hours.get_loc(config.TEST_END)
    times = hours[t0:t1]
    n_r = len(raion_ids)
    X = np.column_stack([features[k][t0:t1].reshape(-1) for k in names] + [np.tile(np.arange(n_r), len(times))])
    row_time = times.repeat(n_r)
    row_raion = np.tile(np.arange(n_r), len(times))

    # AC-2.4: райони з тривогою >95% або <1% часу на train — поза ціллю, але лишаються в ознаках
    train_rows = row_time < config.VAL_START
    share = pd.Series(X[train_rows, names.index("active")]).groupby(row_raion[train_rows]).mean().to_numpy()
    in_target = (share >= TARGET_SHARE[0]) & (share <= TARGET_SHARE[1])
    frontline = share >= np.quantile(share[in_target], FRONTLINE_QUANTILE)
    pd.DataFrame({"raion_id": raion_ids, "train_alert_share": share.round(4), "in_target": in_target, "frontline": frontline}).to_csv(
        config.METRICS / "raions_target.csv", index=False
    )

    rows, predictions = [], []
    for h in config.HORIZONS_H:
        y = active[t0 + h : t1 + h].reshape(-1)
        valid = in_target[row_raion] & ~_in_outage(row_time + timedelta(hours=h))  # AC-1.3
        fit = valid & (row_time < config.TEST_START - timedelta(hours=h))  # проміжок h (AC-5.6)
        test = valid & (row_time >= config.TEST_START)

        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
            categorical_features=[X.shape[1] - 1], early_stopping=False, random_state=0,
        )
        model.fit(X[fit], y[fit])
        p_model = model.predict_proba(X[test])[:, 1]

        # Базові прогнози (AC-6.3); частоти — лише з даних до тесту (AC-5.5)
        hour_col = X[:, names.index("hour")].astype(int)
        clim = pd.Series(y[fit]).groupby([row_raion[fit], hour_col[fit]]).mean()
        p_clim = clim.reindex(pd.MultiIndex.from_arrays([row_raion[test], hour_col[test]])).fillna(y[fit].mean()).to_numpy()
        week_ago = active[t0 + h - 168 : t1 + h - 168].reshape(-1)
        candidates = {
            "model": p_model,
            "persistence": X[test, names.index("active")],
            "week_ago": week_ago[test],
            "climatology": p_clim,
        }

        yt = y[test]
        subsets = {
            "all": np.ones(test.sum(), bool),
            "no_alert_at_T": X[test, names.index("active")] == 0,  # AC-6.4
            "frontline": frontline[row_raion[test]],  # AC-6.7
            "rear": ~frontline[row_raion[test]],
        }
        for model_name, p in candidates.items():
            for subset, m in subsets.items():
                rows.append({"horizon_h": h, "model": model_name, "subset": subset, **_score(yt[m], p[m])})
        predictions.append(
            pd.DataFrame({"time": row_time[test], "raion_id": np.array(raion_ids)[row_raion[test]], "horizon_h": h, "y": yt, "p": p_model})
        )

    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(config.METRICS / "metrics.csv", index=False)
    predictions = pd.concat(predictions)
    predictions.to_parquet(config.FORECASTS / "test_predictions.parquet", index=False)
    export_forecast(predictions, raion_ids, in_target)
    return metrics


def export_forecast(predictions: pd.DataFrame, raion_ids: list[str], in_target: np.ndarray) -> None:
    """Прогноз на останній момент T тестового періоду у форматі контракту (INFORMATION_ARCHITECTURE.md, розділ 7)."""
    as_of = predictions["time"].max()
    last = predictions[predictions["time"] == as_of]
    out = {}
    for raion, ok in zip(raion_ids, in_target):
        if not ok:
            out[raion] = {"status": "not_in_target"}
            continue
        rows = last[last["raion_id"] == raion]
        out[raion] = {"status": "ok", "p": {str(h): round(float(p), 3) for h, p in zip(rows["horizon_h"], rows["p"])}}
    payload = {
        "as_of": as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": "mvp-1",
        "raions": out,
    }
    (config.EXPORT / "forecast.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(result[result["subset"].isin(["all", "no_alert_at_T"])].round(3).to_string(index=False))
