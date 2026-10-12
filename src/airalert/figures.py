"""Графіки результатів для README у WebP без втрат (AC-8.2, AC-8.3, D-12).

Запуск після пайплайну: python -m airalert.figures
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airalert import config  # noqa: E402

# Перевірена палітра (4 категорійні кольори пройшли перевірку на розрізнення
# для дальтонізму; світлим кольорам допомагають підписи значень і таблиця в README)
SURFACE, TEXT, TEXT_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"
SERIES = {"model": "#2a78d6", "climatology": "#eb6834", "persistence": "#1baf7a", "week_ago": "#eda100"}
LABELS = {"model": "Модель", "climatology": "Кліматологія", "persistence": "Як зараз", "week_ago": "Тиждень тому"}
HORIZONS = {1: "#2a78d6", 3: "#eb6834", 6: "#1baf7a"}
KYIV = "Europe/Kyiv"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": TEXT,
    }
)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    fig.savefig(config.FIGURES / name, dpi=200, bbox_inches="tight", pil_kwargs={"lossless": True})
    plt.close(fig)


def brier_by_horizon(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    titles = {"all": "Усі моменти", "no_alert_at_T": "Моменти, коли в районі немає тривоги"}
    width = 0.2
    for ax, (subset, title) in zip(axes, titles.items()):
        data = metrics[metrics["subset"] == subset]
        for i, model in enumerate(SERIES):
            rows = data[data["model"] == model].sort_values("horizon_h")
            x = np.arange(len(rows)) + (i - 1.5) * width
            bars = ax.bar(x, rows["brier"], width, color=SERIES[model], edgecolor=SURFACE, linewidth=1.5, label=LABELS[model])
            ax.bar_label(bars, fmt="%.3f", fontsize=6, color=TEXT_2, padding=2, rotation=90)
        ax.set_xticks(range(3), [f"{h} год" for h in config.HORIZONS_H])
        ax.set_title(title, loc="left")
        _style(ax)
    axes[0].set_ylabel("Brier score (менше — краще)")
    axes[0].set_ylim(0, metrics.loc[metrics["subset"].isin(list(titles)), "brier"].max() * 1.22)  # місце для підписів
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Тест 02.08–29.08.2026: модель проти базових прогнозів", x=0.01, y=1.14, ha="left", fontsize=12)
    _save(fig, "brier_by_horizon.webp")


def calibration(predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Ідеальне калібрування")
    bins = np.linspace(0, 1, 11)
    for h, color in HORIZONS.items():
        d = predictions[predictions["horizon_h"] == h]
        b = np.clip(np.digitize(d["p"], bins) - 1, 0, 9)
        g = d.groupby(b).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        g = g[g["n"] >= 50]
        ax.plot(g["p"], g["y"], color=color, linewidth=1.5, marker="o", markersize=5, markeredgecolor=SURFACE, label=f"{h} год")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Прогнозована ймовірність", ylabel="Фактична частота тривог")
    ax.set_title("Калібрування на тесті", loc="left")
    ax.legend(loc="upper left")
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "calibration.webp")


def timeline(predictions: pd.DataFrame, raion_id: str, raion_name: str) -> None:
    d = predictions[(predictions["horizon_h"] == 1) & (predictions["raion_id"] == raion_id)].copy()
    d["target_time"] = (d["time"] + pd.Timedelta(hours=1)).dt.tz_convert(KYIV)
    d = d[d["target_time"] >= d["target_time"].max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.fill_between(d["target_time"], 0, d["y"], step="post", color=MUTED, alpha=0.6, linewidth=0, label="Фактична тривога")
    ax.plot(d["target_time"], d["p"], color=SERIES["model"], linewidth=1.5, label="Прогноз на 1 год уперед")
    ax.set(ylim=(0, 1.02), ylabel="Ймовірність")
    ax.set_title(f"{raion_name}: останній тиждень тесту (київський час)", loc="left")
    ax.legend(loc="upper left", ncol=2)
    fig.autofmt_xdate()
    _style(ax)
    _save(fig, "timeline_week.webp")


def main() -> None:
    config.ensure_dirs()
    metrics = pd.read_csv(config.METRICS / "metrics.csv")
    predictions = pd.read_parquet(config.FORECASTS / "test_predictions.parquet")
    raions = pd.read_csv(config.REFERENCE / "raions.csv")
    kyiv = raions[raions["oblast_id"] == raions.loc[raions["name_uk"].str.contains("Київ"), "oblast_id"].iloc[0]]
    brier_by_horizon(metrics)
    calibration(predictions)
    kyiv_city = raions[raions["raion_id"].isin(predictions["raion_id"]) & raions["name_uk"].str.fullmatch("Київ|м. Київ")]
    target = kyiv_city.iloc[0] if not kyiv_city.empty else kyiv.iloc[0]
    timeline(predictions, target["raion_id"], target["name_uk"])


if __name__ == "__main__":
    main()
