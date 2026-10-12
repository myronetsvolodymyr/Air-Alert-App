"""Довідники районів і областей з меж COD-AB (крок 2; AC-2.1, AC-6.6).

Запуск: python -m airalert.reference
"""

import numpy as np
import pandas as pd
import pyogrio

from airalert import config, sources
from airalert.names import normalize

GDB = f"/vsizip/{config.COD_AB_RAW}/ukr_admin_boundaries.gdb"
EARTH_RADIUS_KM = 6371.0

# Райони, перейменовані у 2024 р.: у CSV тривог є старі назви, у COD-AB — нові.
# Відповідність звірено з COD-AB у межах тієї самої області.
RENAMED = {
    "красноградський": "берестинський",
    "червоноградський": "шептицький",
    "володимир-волинський": "володимирський",
}


def _key(name: str) -> str:
    """Ключ для зіставлення: без «область»/«район» і префікса «м.», з новими назвами районів."""
    name = normalize(name)
    for suffix in (" область", " район"):
        name = name.removesuffix(suffix)
    name = name.removeprefix("м.").strip()
    return RENAMED.get(name, name)


def load_raions():
    gdf = pyogrio.read_dataframe(
        GDB, layer="ukr_admin2", columns=["adm2_name1", "adm2_pcode", "adm1_name1", "adm1_pcode"]
    ).rename(
        columns={
            "adm2_pcode": "raion_id",
            "adm2_name1": "name_uk",
            "adm1_pcode": "oblast_id",
            "adm1_name1": "oblast_name_uk",
        }
    )
    # Центроїди рахуємо в рівноплощинній проєкції (LAEA Europe), а зберігаємо в градусах
    centroids = gdf.to_crs(3035).centroid.to_crs(4326)
    gdf["lat"] = centroids.y.round(5)
    gdf["lon"] = centroids.x.round(5)
    return gdf.sort_values("raion_id").reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_neighbors(raions) -> pd.DataFrame:
    """Перше кільце — райони зі спільною межею, друге — їхні сусіди (AC-6.6)."""
    shapes = raions[["raion_id", "geometry"]]
    pairs = shapes.sjoin(shapes, predicate="intersects")
    pairs = pairs[pairs["raion_id_left"] != pairs["raion_id_right"]]
    ring1 = pairs.groupby("raion_id_left")["raion_id_right"].apply(set).to_dict()

    rows = []
    for raion, first in ring1.items():
        second = set().union(*(ring1.get(n, set()) for n in first)) - first - {raion}
        rows += [(raion, n, 1) for n in sorted(first)] + [(raion, n, 2) for n in sorted(second)]
    out = pd.DataFrame(rows, columns=["raion_id", "neighbor_id", "ring"])

    coords = raions.set_index("raion_id")[["lat", "lon"]]
    a, b = coords.loc[out["raion_id"]].to_numpy(), coords.loc[out["neighbor_id"]].to_numpy()
    out["distance_km"] = _haversine_km(a[:, 0], a[:, 1], b[:, 0], b[:, 1]).round(1)
    return out


def build_aliases(raions) -> pd.DataFrame:
    """Зводить назви областей і районів з CSV тривог до кодів COD-AB (AC-2.1)."""
    csv = pd.read_csv(config.OFFICIAL_ALERTS_RAW, usecols=["oblast", "raion"]).drop_duplicates()
    oblast_ids = {_key(n): i for n, i in raions[["oblast_name_uk", "oblast_id"]].drop_duplicates().itertuples(index=False)}
    raion_ids = {(o, _key(n)): i for o, n, i in raions[["oblast_id", "name_uk", "raion_id"]].itertuples(index=False)}

    csv["oblast_id"] = csv["oblast"].map(lambda n: oblast_ids.get(_key(n)))
    csv["raion_id"] = [
        raion_ids.get((o, _key(r))) if isinstance(r, str) else None
        for o, r in zip(csv["oblast_id"], csv["raion"])
    ]
    unmatched = csv[csv["oblast_id"].isna() | (csv["raion"].notna() & csv["raion_id"].isna())]
    if not unmatched.empty:
        raise ValueError(f"Не зведено до кодів COD-AB:\n{unmatched.to_string()}")
    return csv.rename(columns={"oblast": "oblast_src", "raion": "raion_src"}).sort_values(
        ["oblast_id", "raion_id"], na_position="first"
    )


def main() -> None:
    config.ensure_dirs()
    sources.download(config.OFFICIAL_ALERTS_URL, config.OFFICIAL_ALERTS_RAW)
    sources.download(config.COD_AB_URL, config.COD_AB_RAW)

    raions = load_raions()
    neighbors = build_neighbors(raions)
    aliases = build_aliases(raions)

    oblasts = raions[["oblast_id", "oblast_name_uk"]].drop_duplicates().rename(columns={"oblast_name_uk": "name_uk"})
    oblasts.to_csv(config.REFERENCE / "oblasts.csv", index=False)
    raions.drop(columns=["geometry", "oblast_name_uk"]).to_csv(config.REFERENCE / "raions.csv", index=False)
    neighbors.to_csv(config.REFERENCE / "raion_neighbors.csv", index=False)
    aliases.to_csv(config.REFERENCE / "name_aliases.csv", index=False)

    area = raions.to_crs(3035).area.sum() / 1e6
    print(f"областей: {len(oblasts)}, районів: {len(raions)}, площа: {area:,.0f} км²")
    print(f"пар сусідів: кільце 1 — {(neighbors['ring'] == 1).sum()}, кільце 2 — {(neighbors['ring'] == 2).sum()}")
    print(f"назв із CSV зведено до кодів: {len(aliases)} (районів: {aliases['raion_id'].nunique()})")


if __name__ == "__main__":
    main()
