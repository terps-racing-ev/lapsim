from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT = Path(__file__).with_name("mis-max-temperature-10-year.png")

years = list(range(2017, 2027))
max_jun_1_5 = [81.9, 81.0, 77.1, 83.5, 84.2, 77.8, 87.0, 82.9, 88.8, 81.3]
max_jun_16_21 = [84.7, 89.0, 77.4, 85.5, 85.2, 93.5, 82.8, 89.4, 88.1, 74.1]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)

fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=False)
fig.patch.set_facecolor("white")

ax.plot(
    years,
    max_jun_1_5,
    color="#2563eb",
    marker="o",
    linewidth=2.6,
    markersize=6.5,
    label="Max daily temperature · June 1–5",
)
ax.plot(
    years,
    max_jun_16_21,
    color="#ea580c",
    marker="o",
    linewidth=2.6,
    markersize=6.5,
    label="Max daily temperature · June 16–21",
)

for x, y in zip(years, max_jun_1_5):
    ax.annotate(f"{y:.1f}°", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", color="#2563eb", fontsize=8)
for x, y in zip(years, max_jun_16_21):
    ax.annotate(f"{y:.1f}°", (x, y), xytext=(0, -16), textcoords="offset points", ha="center", color="#ea580c", fontsize=8)

ax.set_title("Maximum Temperature at Michigan International Speedway", loc="left", fontweight="bold", pad=16)
ax.text(
    0,
    1.015,
    "Highest daily maximum within each June date range · 2017–2026",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    color="#4b5563",
    fontsize=11,
)
ax.set_xlabel("Year")
ax.set_ylabel("Maximum daily temperature (°F)")
ax.set_xticks(years)
ax.set_ylim(70, 98)
ax.set_yticks(range(70, 99, 5))
ax.grid(axis="y", color="#d1d5db", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#9ca3af")
ax.spines["bottom"].set_color("#9ca3af")
ax.tick_params(colors="#374151")
ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.0))

fig.text(
    0.075,
    0.03,
    "Source: Open-Meteo historical weather archive · Track coordinates: 42.0663°N, 84.2414°W",
    fontsize=8.5,
    color="#6b7280",
    ha="left",
)
fig.subplots_adjust(left=0.075, right=0.98, top=0.82, bottom=0.13)
fig.savefig(OUTPUT, dpi=220, facecolor="white")
print(OUTPUT)
