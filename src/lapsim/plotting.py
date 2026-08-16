"""Reusable plots for lap simulation results."""

from typing import TYPE_CHECKING

from utils.units import meters_per_second_to_miles_per_hour

if TYPE_CHECKING:
    from .lap_time_solver import LapResult


STANDARD_GRAVITY_MPS2 = 9.81


def plot_lap_telemetry_summary(result: "LapResult"):
    """Create a four-panel summary from permanent lap telemetry."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    telemetry = result.telemetry
    segment_speed_mph = [
        meters_per_second_to_miles_per_hour(speed_mps)
        for speed_mps in telemetry.speed_mps
    ]
    acceleration_g = [
        acceleration_mps2 / STANDARD_GRAVITY_MPS2
        for acceleration_mps2 in telemetry.longitudinal_acceleration_mps2
    ]
    battery_power_kw = [
        battery_power_w / 1_000.0 for battery_power_w in telemetry.battery_power_w
    ]

    line_segments = [
        [
            (result.x_m[index], result.y_m[index]),
            (result.x_m[index + 1], result.y_m[index + 1]),
        ]
        for index in range(len(result.x_m) - 1)
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor="white")
    figure.suptitle(
        f"Lap-time solution - {result.lap_time_s:.2f} s",
        fontsize=18,
        fontweight="bold",
    )

    map_axes = axes[0, 0]
    speed_normalization = Normalize(
        vmin=min(segment_speed_mph),
        vmax=max(segment_speed_mph),
    )
    colored_track = LineCollection(
        line_segments,
        cmap="turbo",
        norm=speed_normalization,
        linewidth=4.0,
        capstyle="round",
    )
    colored_track.set_array(segment_speed_mph)
    map_axes.add_collection(colored_track)
    map_axes.scatter(
        result.x_m[0],
        result.y_m[0],
        color="black",
        s=55,
        zorder=4,
        label="Start / finish",
    )
    map_axes.autoscale()
    map_axes.set_aspect("equal", adjustable="box")
    map_axes.set_title("Track speed map")
    map_axes.set_xlabel("x [m]")
    map_axes.set_ylabel("y [m]")
    map_axes.legend(loc="best")
    figure.colorbar(
        colored_track,
        ax=map_axes,
        label="Speed [mph]",
        pad=0.02,
    )

    torque_axes = axes[0, 1]
    torque_axes.plot(
        telemetry.distance_m,
        telemetry.motor_torque_nm,
        color="#d1495b",
        linewidth=1.6,
    )
    torque_axes.fill_between(
        telemetry.distance_m,
        telemetry.motor_torque_nm,
        color="#d1495b",
        alpha=0.16,
    )
    torque_axes.set_title("Motor torque")
    torque_axes.set_xlabel("Lap distance [m]")
    torque_axes.set_ylabel("Torque [N m]")
    torque_axes.set_xlim(result.distance_m[0], result.distance_m[-1])
    torque_axes.set_ylim(bottom=0)

    acceleration_axes = axes[1, 0]
    acceleration_axes.plot(
        telemetry.distance_m,
        acceleration_g,
        color="#00798c",
        linewidth=1.35,
    )
    acceleration_axes.fill_between(
        telemetry.distance_m,
        acceleration_g,
        0,
        where=[acceleration >= 0 for acceleration in acceleration_g],
        color="#2a9d8f",
        alpha=0.25,
        label="Acceleration",
    )
    acceleration_axes.fill_between(
        telemetry.distance_m,
        acceleration_g,
        0,
        where=[acceleration < 0 for acceleration in acceleration_g],
        color="#e76f51",
        alpha=0.25,
        label="Braking",
    )
    acceleration_axes.axhline(0, color="0.25", linewidth=0.8)
    acceleration_axes.set_title("Longitudinal acceleration")
    acceleration_axes.set_xlabel("Lap distance [m]")
    acceleration_axes.set_ylabel("Acceleration [g]")
    acceleration_axes.set_xlim(result.distance_m[0], result.distance_m[-1])
    acceleration_axes.legend(loc="best")

    power_axes = axes[1, 1]
    power_line = power_axes.plot(
        telemetry.distance_m,
        battery_power_kw,
        color="#6a4c93",
        linewidth=1.4,
        label="Battery power",
    )[0]
    power_axes.fill_between(
        telemetry.distance_m,
        battery_power_kw,
        color="#6a4c93",
        alpha=0.14,
    )
    power_axes.set_title("Positive battery power and cumulative energy")
    power_axes.set_xlabel("Lap distance [m]")
    power_axes.set_ylabel("Power [kW]", color="#6a4c93")
    power_axes.tick_params(axis="y", labelcolor="#6a4c93")
    power_axes.set_xlim(result.distance_m[0], result.distance_m[-1])
    power_axes.set_ylim(bottom=0)

    energy_axes = power_axes.twinx()
    energy_line = energy_axes.plot(
        telemetry.distance_m,
        telemetry.cumulative_energy_kwh,
        color="#f4a261",
        linewidth=2.0,
        label="Energy used",
    )[0]
    energy_axes.set_ylabel("Energy [kWh]", color="#d97706")
    energy_axes.tick_params(axis="y", labelcolor="#d97706")
    energy_axes.set_ylim(bottom=0)
    power_axes.legend(
        [power_line, energy_line],
        ["Battery power", "Energy used"],
        loc="upper left",
    )

    figure.text(
        0.5,
        0.015,
        f"Positive electrical energy: {telemetry.total_energy_kwh:.3f} kWh"
        f"  |  Starting speed: "
        f"{meters_per_second_to_miles_per_hour(result.starting_speed_mps):.1f} mph"
        f"  |  No regenerative braking or auxiliary loads",
        ha="center",
        fontsize=10,
        color="0.3",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    return figure, axes
