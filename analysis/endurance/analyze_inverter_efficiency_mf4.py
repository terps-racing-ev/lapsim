"""Estimate inverter-plus-motor efficiency from an endurance MF4 log.

The log does not contain a trustworthy three-phase real-power channel, so an
inverter-only efficiency cannot be calculated from it. This analysis compares
HVC pack electrical power with motor shaft power (torque feedback * speed) and
therefore measures the combined inverter-plus-motor conversion efficiency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from asammdf import MDF
except ImportError as exc:  # pragma: no cover - depends on the optional MF4 extra
    raise ImportError("asammdf is required: pip install -e '.[mf4]'") from exc


ROOT = Path(__file__).resolve().parents[2]
INVERTER_DBC = ROOT / "logs" / "Inverter.dbc"
HVC_DBC = ROOT / "logs" / "hvc (1).dbc"

# Established by aligning HVC pack power with torque * speed on this recording.
DEFAULT_HVC_POWER_LAG_S = 0.09
DEFAULT_SAMPLE_PERIOD_S = 0.1
DEFAULT_MIN_POWER_W = 1_000.0
DEFAULT_MAX_EFFICIENCY = 1.0

SIGNAL_CANDIDATES = {
    "torque_nm": (
        "CAN1.High_Speed_Message.HS_Torque_Feedback",
        "High_Speed_Message.HS_Torque_Feedback",
        "HS_Torque_Feedback",
        "CAN1.Torque_Timer_Info.INV_Torque_Feedback",
        "Torque_Timer_Info.INV_Torque_Feedback",
        "INV_Torque_Feedback",
    ),
    "motor_speed_rpm": (
        "CAN1.High_Speed_Message.HS_Motor_Speed",
        "High_Speed_Message.HS_Motor_Speed",
        "HS_Motor_Speed",
        "CAN1.Motor_Position_Info.INV_Motor_Speed",
        "Motor_Position_Info.INV_Motor_Speed",
        "INV_Motor_Speed",
    ),
    "pack_power_kw": (
        "CAN1.HVC_IO_Summary.HVC_Pack_Power_kW",
        "HVC_IO_Summary.HVC_Pack_Power_kW",
        "HVC_Pack_Power_kW",
    ),
    "pack_current_a": (
        "CAN1.HVC_IO_Current.HVC_Pack_Current_A",
        "HVC_IO_Current.HVC_Pack_Current_A",
        "HVC_Pack_Current_A",
    ),
    "pack_voltage_v": (
        "CAN1.HVC_IO_VSense.HVC_Batt_Voltage_V",
        "HVC_IO_VSense.HVC_Batt_Voltage_V",
        "HVC_Batt_Voltage_V",
    ),
    "dc_bus_current_a": (
        "CAN1.Current_Info.INV_DC_Bus_Current",
        "Current_Info.INV_DC_Bus_Current",
        "INV_DC_Bus_Current",
    ),
    "dc_bus_voltage_v": (
        "CAN1.High_Speed_Message.HS_DC_Bus_Voltage",
        "High_Speed_Message.HS_DC_Bus_Voltage",
        "HS_DC_Bus_Voltage",
        "CAN1.Voltage_Info.INV_DC_Bus_Voltage",
        "Voltage_Info.INV_DC_Bus_Voltage",
        "INV_DC_Bus_Voltage",
    ),
}


@dataclass(frozen=True)
class SignalData:
    channel: str
    time_s: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class EfficiencyResult:
    efficiency: np.ndarray
    loss_w: np.ndarray
    motoring: np.ndarray
    regenerating: np.ndarray
    valid: np.ndarray
    directionally_aligned: np.ndarray
    energy_weighted_efficiency: float
    motoring_energy_weighted_efficiency: float
    regenerating_energy_weighted_efficiency: float


def get_signal(decoded: MDF, key: str, *, required: bool = False) -> SignalData | None:
    """Return the first available numeric channel for a logical signal."""

    for candidate in SIGNAL_CANDIDATES[key]:
        if candidate not in decoded.channels_db:
            continue
        signal = decoded.get(candidate)
        values = np.asarray(signal.samples, dtype=float)
        time_s = np.asarray(signal.timestamps, dtype=float)
        finite_time = np.isfinite(time_s)
        if finite_time.sum() >= 2:
            return SignalData(candidate, time_s[finite_time], values[finite_time])
    if required:
        raise KeyError(
            f"No channel found for {key}. Tried: "
            + ", ".join(SIGNAL_CANDIDATES[key])
        )
    return None


def interpolate(signal: SignalData, query_time_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate without extending beyond the recorded channel."""

    finite = np.isfinite(signal.time_s) & np.isfinite(signal.values)
    if finite.sum() < 2:
        return np.full(len(query_time_s), np.nan)
    return np.interp(
        query_time_s,
        signal.time_s[finite],
        signal.values[finite],
        left=np.nan,
        right=np.nan,
    )


def common_clock(
    signals: tuple[SignalData, ...],
    *,
    sample_period_s: float,
    electrical_lag_s: float,
) -> np.ndarray:
    """Build a uniform clock over the common, lag-corrected signal interval."""

    if not np.isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample period must be finite and positive")
    starts = [signals[0].time_s[0], signals[1].time_s[0]]
    finishes = [signals[0].time_s[-1], signals[1].time_s[-1]]
    starts.extend(signal.time_s[0] - electrical_lag_s for signal in signals[2:])
    finishes.extend(signal.time_s[-1] - electrical_lag_s for signal in signals[2:])
    start_s = max(starts)
    finish_s = min(finishes)
    if not np.isfinite(start_s) or not np.isfinite(finish_s) or finish_s <= start_s:
        raise ValueError("Required MF4 signals do not have an overlapping time range")
    start_s = np.ceil(start_s / sample_period_s) * sample_period_s
    return np.arange(start_s, finish_s + sample_period_s * 0.25, sample_period_s)


def calculate_efficiency(
    dc_power_w: np.ndarray,
    shaft_power_w: np.ndarray,
    *,
    min_power_w: float,
    max_efficiency: float = DEFAULT_MAX_EFFICIENCY,
) -> EfficiencyResult:
    """Calculate physically directed motoring and regeneration efficiency.

    Motoring is ``shaft / DC``. Regeneration reverses the energy flow and is
    ``DC / shaft`` (equivalently ``abs(DC) / abs(shaft)``). Near-zero power,
    opposite-sign samples, and physically impossible ratios above one are not measurements
    of steady conversion efficiency and are left as NaN.
    """

    if dc_power_w.shape != shaft_power_w.shape:
        raise ValueError("DC and shaft power arrays must have the same shape")
    if not np.isfinite(min_power_w) or min_power_w < 0.0:
        raise ValueError("minimum power must be finite and nonnegative")
    if not np.isfinite(max_efficiency) or max_efficiency <= 0.0:
        raise ValueError("maximum efficiency must be finite and positive")

    finite = np.isfinite(dc_power_w) & np.isfinite(shaft_power_w)
    motoring_candidates = finite & (dc_power_w >= min_power_w) & (
        shaft_power_w >= min_power_w
    )
    regenerating_candidates = finite & (dc_power_w <= -min_power_w) & (
        shaft_power_w <= -min_power_w
    )
    raw_efficiency = np.full(dc_power_w.shape, np.nan, dtype=float)
    raw_efficiency[motoring_candidates] = (
        shaft_power_w[motoring_candidates] / dc_power_w[motoring_candidates]
    )
    raw_efficiency[regenerating_candidates] = (
        dc_power_w[regenerating_candidates] / shaft_power_w[regenerating_candidates]
    )
    directionally_aligned = motoring_candidates | regenerating_candidates
    valid = directionally_aligned & (raw_efficiency > 0.0) & (
        raw_efficiency <= max_efficiency
    )
    motoring = motoring_candidates & valid
    regenerating = regenerating_candidates & valid

    efficiency = np.where(valid, raw_efficiency, np.nan)
    input_power_w = np.where(
        motoring, dc_power_w, np.where(regenerating, -shaft_power_w, np.nan)
    )
    output_power_w = np.where(
        motoring, shaft_power_w, np.where(regenerating, -dc_power_w, np.nan)
    )
    loss_w = np.where(valid, input_power_w - output_power_w, np.nan)
    # Do not censor samples above 100% from the energy totals: those samples
    # arise around fast transients because channels have different sample rates.
    # Censoring only their positive error would bias the aggregate downward.
    motoring_input_energy = float(np.sum(dc_power_w[motoring_candidates]))
    motoring_output_energy = float(np.sum(shaft_power_w[motoring_candidates]))
    regen_input_energy = float(np.sum(-shaft_power_w[regenerating_candidates]))
    regen_output_energy = float(np.sum(-dc_power_w[regenerating_candidates]))
    input_energy = motoring_input_energy + regen_input_energy
    output_energy = motoring_output_energy + regen_output_energy
    energy_weighted_efficiency = (
        output_energy / input_energy if input_energy > 0.0 else float("nan")
    )
    motoring_energy_weighted_efficiency = (
        motoring_output_energy / motoring_input_energy
        if motoring_input_energy > 0.0
        else float("nan")
    )
    regenerating_energy_weighted_efficiency = (
        regen_output_energy / regen_input_energy
        if regen_input_energy > 0.0
        else float("nan")
    )
    return EfficiencyResult(
        efficiency,
        loss_w,
        motoring,
        regenerating,
        valid,
        directionally_aligned,
        energy_weighted_efficiency,
        motoring_energy_weighted_efficiency,
        regenerating_energy_weighted_efficiency,
    )


def decode_required_signals(mf4_path: Path) -> tuple[MDF, dict[str, SignalData | None]]:
    if not mf4_path.is_file():
        raise FileNotFoundError(f"MF4 not found: {mf4_path}")
    missing_dbcs = [path for path in (INVERTER_DBC, HVC_DBC) if not path.is_file()]
    if missing_dbcs:
        raise FileNotFoundError("Missing DBC file(s): " + ", ".join(map(str, missing_dbcs)))

    decoded = MDF(mf4_path).extract_bus_logging(
        {"CAN": [(str(INVERTER_DBC), 1), (str(HVC_DBC), 1)]}
    )
    signals = {
        key: get_signal(decoded, key, required=key in {"torque_nm", "motor_speed_rpm"})
        for key in SIGNAL_CANDIDATES
    }
    return decoded, signals


def get_signal_by_suffix(decoded: MDF, suffix: str) -> SignalData | None:
    """Resolve an optional plotting channel without adding it to core inputs."""

    candidates = sorted(
        (name for name in decoded.channels_db if name == suffix or name.endswith(f".{suffix}")),
        key=lambda name: (not name.startswith("CAN1."), len(name)),
    )
    for candidate in candidates:
        signal = decoded.get(candidate)
        time_s = np.asarray(signal.timestamps, dtype=float)
        values = np.asarray(signal.samples, dtype=float)
        if len(time_s) >= 2:
            return SignalData(candidate, time_s, values)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mf4", type=Path, help="Endurance MF4 log file")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--sample-period-s", type=float, default=DEFAULT_SAMPLE_PERIOD_S)
    parser.add_argument(
        "--hvc-power-lag-s",
        type=float,
        default=DEFAULT_HVC_POWER_LAG_S,
        help="HVC timestamp lag to remove (default: %(default)s s)",
    )
    parser.add_argument(
        "--min-power-w",
        type=float,
        default=DEFAULT_MIN_POWER_W,
        help="Minimum input and output magnitude used for efficiency (default: %(default)s W)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decoded, signals = decode_required_signals(args.mf4)
    torque = signals["torque_nm"]
    speed = signals["motor_speed_rpm"]
    assert torque is not None and speed is not None

    pack_power = signals["pack_power_kw"]
    pack_current = signals["pack_current_a"]
    pack_voltage = signals["pack_voltage_v"]
    dc_current = signals["dc_bus_current_a"]
    dc_voltage = signals["dc_bus_voltage_v"]

    if pack_power is not None:
        electrical_signals = (pack_power,)
        electrical_lag_s = args.hvc_power_lag_s
        dc_source = pack_power.channel
    elif pack_current is not None and pack_voltage is not None:
        electrical_signals = (pack_current, pack_voltage)
        electrical_lag_s = args.hvc_power_lag_s
        dc_source = f"{pack_current.channel} * {pack_voltage.channel}"
    elif dc_current is not None and dc_voltage is not None:
        electrical_signals = (dc_current, dc_voltage)
        electrical_lag_s = 0.0
        dc_source = f"{dc_current.channel} * {dc_voltage.channel}"
    else:
        decoded.close()
        raise KeyError(
            "No DC power source: need HVC pack power, HVC current + voltage, "
            "or inverter DC-bus current + voltage"
        )

    mf4_time_s = common_clock(
        (torque, speed, *electrical_signals),
        sample_period_s=args.sample_period_s,
        electrical_lag_s=electrical_lag_s,
    )
    query_electrical_time_s = mf4_time_s + electrical_lag_s
    torque_nm = interpolate(torque, mf4_time_s)
    speed_rpm = interpolate(speed, mf4_time_s)
    shaft_power_w = torque_nm * speed_rpm * (2.0 * np.pi / 60.0)

    if pack_power is not None:
        dc_power_w = interpolate(pack_power, query_electrical_time_s) * 1_000.0
    elif pack_current is not None and pack_voltage is not None:
        dc_power_w = interpolate(pack_current, query_electrical_time_s) * interpolate(
            pack_voltage, query_electrical_time_s
        )
    else:
        assert dc_current is not None and dc_voltage is not None
        dc_power_w = interpolate(dc_current, query_electrical_time_s) * interpolate(
            dc_voltage, query_electrical_time_s
        )

    result = calculate_efficiency(
        dc_power_w,
        shaft_power_w,
        min_power_w=args.min_power_w,
    )
    if not result.valid.any():
        decoded.close()
        raise RuntimeError(
            "No valid efficiency samples remain after power, direction, and plausibility filters"
        )

    time_s = mf4_time_s - mf4_time_s[0]
    instantaneous = result.efficiency[result.valid]
    mean_instantaneous = float(np.mean(instantaneous))
    median_instantaneous = float(np.median(instantaneous))
    mean_loss_w = float(np.nanmean(result.loss_w))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, layout="constrained")
    axes[0].plot(time_s, dc_power_w / 1_000.0, color="#4C78A8", lw=1.0, label="Pack DC power")
    axes[0].plot(
        time_s,
        shaft_power_w / 1_000.0,
        color="#F58518",
        lw=1.0,
        label="Motor shaft power",
    )
    axes[0].set_ylabel("Power [kW]")
    axes[0].set_title("Electrical and mechanical power (positive = motoring)")
    axes[0].legend(loc="upper right", frameon=False)

    motoring_efficiency = np.where(result.motoring, result.efficiency * 100.0, np.nan)
    regen_efficiency = np.where(result.regenerating, result.efficiency * 100.0, np.nan)
    axes[1].plot(time_s, motoring_efficiency, color="#54A24B", lw=1.0, label="Motoring")
    axes[1].plot(time_s, regen_efficiency, color="#B279A2", lw=1.0, label="Regeneration")
    axes[1].axhline(
        result.energy_weighted_efficiency * 100.0,
        color="#333333",
        ls="--",
        lw=0.9,
        label=f"Energy-weighted = {result.energy_weighted_efficiency:.1%}",
    )
    axes[1].set_ylabel("Efficiency [%]")
    axes[1].set_ylim(0.0, DEFAULT_MAX_EFFICIENCY * 100.0)
    axes[1].set_title("Valid conversion-efficiency samples")
    axes[1].legend(loc="lower right", frameon=False)

    axes[2].plot(time_s, result.loss_w / 1_000.0, color="#E45756", lw=0.9)
    axes[2].fill_between(time_s, result.loss_w / 1_000.0, color="#E45756", alpha=0.3)
    axes[2].set_ylabel("Conversion loss [kW]")
    axes[2].set_xlabel("Time since common signal start [s]")
    axes[2].set_title("Input power minus output power on valid samples")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Observed inverter + motor efficiency from MF4  |  "
        f"energy-weighted={result.energy_weighted_efficiency:.1%}  |  "
        f"valid samples={int(result.valid.sum()):,}",
        fontsize=12,
    )
    plot_path = args.output_dir / "inverter_efficiency_endurance.png"
    fig.savefig(plot_path, dpi=190)
    plt.close(fig)

    phase_currents = [
        signal
        for suffix in ("INV_Phase_A_Current", "INV_Phase_B_Current", "INV_Phase_C_Current")
        if (signal := get_signal_by_suffix(decoded, suffix)) is not None
    ]
    if phase_currents:
        fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
        for signal in phase_currents:
            ax.plot(
                signal.time_s - mf4_time_s[0],
                signal.values,
                label=signal.channel.split(".")[-1],
                lw=0.8,
            )
        ax.set_title("Inverter phase currents")
        ax.set_ylabel("Current [A]")
        ax.set_xlabel("Time since common signal start [s]")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        phase_path = args.output_dir / "inverter_phase_currents.png"
        fig.savefig(phase_path, dpi=190)
        plt.close(fig)
        print(f"Saved phase-current plot: {phase_path}")

    summary_path = args.output_dir / "inverter_efficiency_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(
            [
                ["measurement", "inverter_plus_motor_dc_to_shaft"],
                ["dc_power_source", dc_source],
                ["hvc_power_lag_s", f"{electrical_lag_s:.6f}"],
                ["minimum_power_w", f"{args.min_power_w:.2f}"],
                ["energy_weighted_efficiency", f"{result.energy_weighted_efficiency:.6f}"],
                [
                    "energy_weighted_efficiency_percent",
                    f"{result.energy_weighted_efficiency * 100.0:.3f}",
                ],
                [
                    "motoring_energy_weighted_efficiency",
                    f"{result.motoring_energy_weighted_efficiency:.6f}",
                ],
                [
                    "regenerating_energy_weighted_efficiency",
                    f"{result.regenerating_energy_weighted_efficiency:.6f}",
                ],
                ["mean_instantaneous_efficiency", f"{mean_instantaneous:.6f}"],
                ["median_instantaneous_efficiency", f"{median_instantaneous:.6f}"],
                ["mean_conversion_loss_w", f"{mean_loss_w:.2f}"],
                ["valid_samples", int(result.valid.sum())],
                ["motoring_samples", int(result.motoring.sum())],
                ["regenerating_samples", int(result.regenerating.sum())],
                ["directionally_aligned_samples", int(result.directionally_aligned.sum())],
                ["total_resampled_samples", len(mf4_time_s)],
            ]
        )
    decoded.close()

    print(f"DC power source: {dc_source}")
    print(f"Applied electrical-power lag correction: {electrical_lag_s:.3f} s")
    print(f"Energy-weighted inverter + motor efficiency: {result.energy_weighted_efficiency:.2%}")
    print(f"Saved efficiency plot: {plot_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
