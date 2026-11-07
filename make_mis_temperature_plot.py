from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("mis-temperature-june.png")

data = {
    "June 1–5": {
        "dates": ["Jun 1", "Jun 2", "Jun 3", "Jun 4", "Jun 5"],
        "2024": [64.1, 66.5, 68.9, 72.3, 71.9],
        "2025": [55.8, 63.8, 74.6, 74.5, 62.4],
        "2026": [64.1, 62.9, 66.3, 70.6, 71.0],
    },
    "June 16–21": {
        "dates": ["Jun 16", "Jun 17", "Jun 18", "Jun 19", "Jun 20", "Jun 21"],
        "2024": [72.9, 80.9, 80.3, 80.0, 80.0, 78.1],
        "2025": [70.7, 75.1, 74.0, 67.7, 70.9, 78.5],
        "2026": [64.2, 61.9, 63.7, 63.6, 64.8, 63.1],
    },
}

colors = {"2024": "#2563eb", "2025": "#ea580c", "2026": "#16a34a"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True, constrained_layout=False)
fig.patch.set_facecolor("white")

for ax, (window, series) in zip(axes, data.items()):
    x = range(len(series["dates"]))
    for year in ("2024", "2025", "2026"):
        ax.plot(
            x,
            series[year],
            marker="o",
            linewidth=2.5,
            markersize=6,
            color=colors[year],
            label=year,
        )

    ax.set_title(window, loc="left", fontweight="bold", pad=12)
    ax.set_xticks(list(x), series["dates"])
    ax.set_ylim(50, 85)
    ax.set_yticks(range(50, 86, 5))
    ax.set_ylabel("Daily mean temperature (°F)")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors="#374151")

    for year in ("2024", "2025", "2026"):
        average = sum(series[year]) / len(series[year])
        ax.text(
            0.99,
            0.05 + (int(year) - 2024) * 0.075,
            f"{year} avg: {average:.1f}°F",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=colors[year],
            fontsize=9,
        )

axes[0].legend(loc="upper left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.01))
fig.suptitle(
    "Average Temperature at Michigan International Speedway",
    fontsize=20,
    fontweight="bold",
    x=0.05,
    ha="left",
    y=0.98,
)
fig.text(
    0.05,
    0.935,
    "Daily mean air temperature at 2 m · Recent June windows · 2024–2026",
    fontsize=11,
    color="#4b5563",
    ha="left",
)
fig.text(
    0.05,
    0.03,
    "Source: Open-Meteo historical weather archive · Track coordinates: 42.0663°N, 84.2414°W",
    fontsize=8.5,
    color="#6b7280",
    ha="left",
)
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.13, wspace=0.14)
fig.savefig(OUTPUT, dpi=220, facecolor="white")
print(OUTPUT)
