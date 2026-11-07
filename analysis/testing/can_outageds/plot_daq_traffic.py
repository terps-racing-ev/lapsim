"""Estimate CAN message traffic from MF4 signals with DBF_/DBL_/DBR_ names."""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from asammdf import MDF


START, END = 884.0, 894.0
BIN_SECONDS = 0.25
# Extended Classical CAN, 8 data bytes: 39 header + 64 data + 16 CRC
# + 9 footer + 3 intermission bits. Dynamic stuffing is unknown from
# decoded signals; the worst-case calculation adds 29 stuff bits.
BITS_PER_FRAME = 131
WORST_CASE_BITS_PER_FRAME = 160
PREFIXES = ("DBF_", "DBL_", "DBR_")
COLORS = {"DBF_": "#277DA1", "DBL_": "#F4A261", "DBR_": "#B85C8A"}
ROOT = Path(__file__).resolve().parent


def main() -> None:
    mf4_path = next(ROOT.glob("*.mf4"))
    rows = []
    with MDF(mf4_path) as mdf:
        for group_index, group in enumerate(mdf.groups):
            signal_names = [
                channel.name
                for channel in group.channels[1:]
                if channel.name.startswith(PREFIXES)
            ]
            if not signal_names:
                continue
            timestamps = mdf.get("time", group=group_index, index=0).samples
            timestamps = timestamps[(timestamps >= START) & (timestamps <= END)]
            if not len(timestamps):
                continue
            prefix = signal_names[0][:4]
            rows.append(
                (
                    prefix,
                    group.channel_group.acq_source.name,
                    group.channel_group.acq_name.split()[0],
                    timestamps,
                    len(signal_names),
                )
            )

    rows.sort(key=lambda row: (PREFIXES.index(row[0]), row[1]))
    edges = np.arange(START, END + BIN_SECONDS / 2, BIN_SECONDS)
    counts = defaultdict(lambda: np.zeros(len(edges) - 1, dtype=int))
    totals = defaultdict(int)
    for prefix, _, _, timestamps, _ in rows:
        counts[prefix] += np.histogram(timestamps, bins=edges)[0]
        totals[prefix] += len(timestamps)

    fig = plt.figure(figsize=(15, 13), layout="constrained")
    axes = fig.subplots(3, 1, height_ratios=[1.7, 1.7, 3.5], sharex=True)
    fig.suptitle(
        "Estimated DAQ CAN traffic, 884–894 s",
        fontsize=18,
        fontweight="bold",
        ha="left",
        x=0.12,
    )
    ax_rate, ax_bits, ax_raster = axes
    bottom = np.zeros(len(edges) - 1)
    for prefix in PREFIXES:
        rate = counts[prefix] / BIN_SECONDS
        ax_rate.bar(
            edges[:-1],
            rate,
            width=BIN_SECONDS,
            align="edge",
            bottom=bottom,
            color=COLORS[prefix],
            label=f"{prefix[:-1]}  ({totals[prefix]} messages)",
        )
        bottom += rate
    ax_rate.set_ylabel("Estimated messages/s\n(0.25 s bins)")
    ax_rate.set_ylim(bottom=0)
    ax_rate.grid(axis="y", alpha=0.25)
    ax_rate.set_axisbelow(True)
    ax_rate.legend(loc="upper left", ncol=3, frameon=False)
    ax_rate.text(
        0.99,
        0.97,
        f"Total: {sum(totals.values()):,} messages  ·  average: {sum(totals.values()) / (END - START):.1f}/s",
        transform=ax_rate.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    bottom_bits = np.zeros(len(edges) - 1)
    for prefix in PREFIXES:
        bit_rate = counts[prefix] * BITS_PER_FRAME / BIN_SECONDS
        ax_bits.bar(
            edges[:-1],
            bit_rate,
            width=BIN_SECONDS,
            align="edge",
            bottom=bottom_bits,
            color=COLORS[prefix],
        )
        bottom_bits += bit_rate
    worst_case = sum(counts[prefix] for prefix in PREFIXES) * WORST_CASE_BITS_PER_FRAME / BIN_SECONDS
    ax_bits.stairs(
        worst_case,
        edges,
        color="#303030",
        linewidth=1.3,
        linestyle="--",
        label="Worst-case bit stuffing",
    )
    ax_bits.set_ylabel("Estimated bit/s\n(0.25 s bins)")
    ax_bits.set_ylim(bottom=0)
    ax_bits.grid(axis="y", alpha=0.25)
    ax_bits.set_axisbelow(True)
    ax_bits.legend(loc="upper left", frameon=False)
    ax_bits.text(
        0.99,
        0.97,
        f"131 bits/frame baseline · {sum(totals.values()) * BITS_PER_FRAME / (END - START):,.0f} bit/s average",
        transform=ax_bits.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )

    for row_number, (prefix, message, bus, timestamps, _) in enumerate(rows):
        ax_raster.scatter(
            timestamps,
            np.full(len(timestamps), row_number),
            marker="|",
            s=38,
            linewidths=0.8,
            color=COLORS[prefix],
        )
    ax_raster.set_yticks(range(len(rows)), [f"{prefix[:-1]}  {message}" for prefix, message, *_ in rows])
    ax_raster.invert_yaxis()
    ax_raster.set_xlim(START, END)
    ax_raster.set_xticks(np.arange(START, END + 1, 1))
    ax_raster.set_xlabel("MF4 elapsed time (s)")
    ax_raster.set_ylabel("Decoded message group")
    ax_raster.grid(axis="x", alpha=0.25)
    ax_raster.set_axisbelow(True)
    ax_raster.set_title("Each tick is one timestamp shared by that message's matching signals", loc="left", fontsize=11)
    for boundary in (9, 16):
        ax_raster.axhline(boundary - 0.5, color="#A9A9A9", linewidth=0.8)

    output = ROOT / "daq_traffic_884-894s.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Matched {sum(row[4] for row in rows)} signals across {len(rows)} message groups on {sorted(set(row[2] for row in rows))}")
    for prefix in PREFIXES:
        print(f"{prefix}: {totals[prefix]} messages")
    print("Total messages:", sum(totals.values()))
    print("Estimated average bit/s (no stuffing):", sum(totals.values()) * BITS_PER_FRAME / (END - START))


if __name__ == "__main__":
    main()
