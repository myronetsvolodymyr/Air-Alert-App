"""Шляхи, джерела й часові межі проєкту.

Значення відповідають INFORMATION_ARCHITECTURE.md (розділи 5–6) і рішенням
D-1…D-7 у JOB_STORIES.md. Змінюємо їх тут, а не в коді кроків.
"""

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REFERENCE = DATA / "reference"

OUTPUTS = ROOT / "outputs"
FORECASTS = OUTPUTS / "forecasts"
METRICS = OUTPUTS / "metrics"
EXPORT = OUTPUTS / "export"

FIGURES = ROOT / "reports" / "figures"


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


# Джерела (додаток A в JOB_STORIES.md)
OFFICIAL_ALERTS_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/official_data_uk.csv"
)
OFFICIAL_ALERTS_RAW = RAW / "official_data_uk.csv"

# Межі адмінодиниць OCHA COD-AB Ukraine (HDX, ліцензія CC BY-IGO)
COD_AB_URL = (
    "https://data.humdata.org/dataset/d23f529f-31e4-4021-a65b-13987e5cfb42/resource/"
    "51ca7001-908b-4f57-be58-e85e10c6d3c4/download/ukr_admin_boundaries.gdb.zip"
)
COD_AB_RAW = RAW / "cod_ab" / "ukr_admin_boundaries.gdb.zip"

# Часова модель MVP: горизонти (D-1) і межі вибірок (D-5, D-7).
# Кінцеві межі невключні; між вибірками ще лишаємо проміжок h годин (AC-5.6).
HORIZONS_H = (1, 3, 6)
TRAIN_START = utc(2025, 11, 1)
VAL_START = utc(2026, 7, 5)
TEST_START = utc(2026, 8, 2)
TEST_END = utc(2026, 8, 30)

# Періоди, коли джерело не працювало: пропуск, а не «тиша» (AC-1.3)
OUTAGES = (
    (utc(2026, 8, 30), utc(2026, 9, 2), "official_csv: жодного запису по країні"),
)


def ensure_dirs() -> None:
    for path in (RAW, INTERIM, PROCESSED, REFERENCE, FORECASTS, METRICS, EXPORT, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
