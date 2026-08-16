r"""Shared utilities and CLI for converting selected MF4 signals to CSV.

The converter exports only an explicit signal profile, not every MF4 channel.
By default it converts the selected endurance-analysis signals for the full
recording on the native GNSS-speed clock. Output rows can instead use an
explicit time window or uniform frequency.

Example::

    ..\.tmp_mf4\venv\Scripts\python.exe analysis\mf4_to_csv\convert_mf4_to_csv.py `
      logs\6.20_endurance_comp.MF4
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import csv
import json

from asammdf import MDF
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
DEFAULT_DBC_DIR = PROJECT_ROOT.parents[2] / "Uploader" / "dbc"
DEFAULT_MF4 = PROJECT_ROOT / "logs" / "6.20_endurance_comp.MF4"
DEFAULT_OUTPUT = ANALYSIS_DIR / "endurance_selected.csv"
DEFAULT_CLOCK_CHANNEL = "CAN9.GnssSpeed.Speed"
DEFAULT_HVC_POWER_LAG_S = 0.09
DBC_FILES = {
    "canedge": "can9-database-01.09.dbc",
    "vcu": "VCU.dbc",
    "mobo": "Baby_MOBO.dbc",
    "inverter": "Inverter.dbc",
    "hvc": "hvc.dbc",
}


@dataclass(frozen=True, slots=True)
class SignalSpec:
    output_name: str
    candidates: tuple[str, ...]
    sampling: Literal["linear", "hold"] = "linear"
    required: bool = True
    backward_time_shift_s: float = 0.0


DEFAULT_SELECTED_SIGNALS = (
    SignalSpec("gps_latitude_deg", ("CAN9.GnssPos.Latitude",)),
    SignalSpec("gps_longitude_deg", ("CAN9.GnssPos.Longitude",)),
    SignalSpec("gps_distance_trip_m", ("CAN9.GnssDistance.DistanceTrip",)),
    SignalSpec("accel_x_mps2", ("CAN9.ImuData.AccelerationX",)),
    SignalSpec("accel_y_mps2", ("CAN9.ImuData.AccelerationY",)),
    SignalSpec("accel_z_mps2", ("CAN9.ImuData.AccelerationZ",)),
    SignalSpec("angular_rate_x_degps", ("CAN9.ImuData.AngularRateX",)),
    SignalSpec("angular_rate_y_degps", ("CAN9.ImuData.AngularRateY",)),
    SignalSpec("angular_rate_z_degps", ("CAN9.ImuData.AngularRateZ",)),
    SignalSpec("motor_rpm", ("CAN1.High_Speed_Message.HS_Motor_Speed",)),
    SignalSpec(
        "torque_command_nm",
        (
            "CAN1.High_Speed_Message.HS_Torque_Command",
            "CAN1.VCU_INV_Command.VCU_INV_Torque_Cmd",
        ),
        sampling="hold",
    ),
    SignalSpec(
        "torque_feedback_nm",
        (
            "CAN1.High_Speed_Message.HS_Torque_Feedback",
            "CAN1.Torque_Timer_Info.INV_Torque_Feedback",
        ),
    ),
    SignalSpec(
        "apps_percent",
        ("CAN1.VCU_APPS_Values.VCU_APPS_Value",),
        sampling="hold",
    ),
    SignalSpec(
        "brake_pressure_psi",
        ("CAN1.VCU_BSE.VCU_BSE_PSI",),
        sampling="hold",
    ),
    SignalSpec(
        "front_brake_pressure_psi",
        ("CAN1.VCU_BSE.VCU_BSE_PSI",),
        sampling="hold",
    ),
    SignalSpec(
        "rear_brake_pressure_psi",
        ("CAN1.MOBO_Power_Telemetry.MOBO_BSE_PSI_Rear",),
        sampling="linear",
    ),
    SignalSpec(
        "battery_power_kw",
        ("CAN1.HVC_IO_Summary.HVC_Pack_Power_kW",),
        backward_time_shift_s=DEFAULT_HVC_POWER_LAG_S,
    ),
    SignalSpec("battery_voltage_v", ("CAN1.HVC_IO_VSense.HVC_Batt_Voltage_V",)),
    SignalSpec("battery_current_a", ("CAN1.HVC_IO_Current.HVC_Pack_Current_A",)),
    SignalSpec(
        "battery_soc_percent",
        ("CAN1.HVC_SOC.HVC_SOC_Percent",),
        sampling="hold",
    ),
)


def decode_mf4(mf4: Path, dbc_dir: Path = DEFAULT_DBC_DIR) -> MDF:
    """Decode the supported CAN buses using the team DBC files."""

    if not mf4.is_file():
        raise FileNotFoundError(mf4)
    dbc_paths = {name: dbc_dir / filename for name, filename in DBC_FILES.items()}
    missing = [str(path) for path in dbc_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DBC file(s):\n" + "\n".join(missing))
    return MDF(mf4).extract_bus_logging(
        {
            "CAN": [
                (dbc_paths["canedge"], 9),
                (dbc_paths["vcu"], 1),
                (dbc_paths["mobo"], 1),
                (dbc_paths["inverter"], 1),
                (dbc_paths["hvc"], 1),
            ]
        }
    )


def resolve_channel(decoded: MDF, spec: SignalSpec) -> str | None:
    for candidate in spec.candidates:
        if candidate in decoded.channels_db:
            return candidate
    if spec.required:
        raise KeyError(f"No decoded channel for {spec.output_name}: {spec.candidates}")
    return None


def signal_arrays(decoded: MDF, channel_name: str) -> tuple[np.ndarray, np.ndarray]:
    signal = decoded.get(channel_name)
    samples = np.asarray(signal.samples)
    if not np.issubdtype(samples.dtype, np.number):
        raise TypeError(f"Channel {channel_name} is not numeric")
    return (
        np.asarray(signal.timestamps, dtype=float),
        np.asarray(samples, dtype=float),
    )


def source_rate_hz(time_s: np.ndarray) -> float | None:
    intervals_s = np.diff(time_s[np.isfinite(time_s)])
    intervals_s = intervals_s[intervals_s > 0.0]
    if not len(intervals_s):
        return None
    return float(1.0 / np.median(intervals_s))


def interpolate(
    source_time_s: np.ndarray,
    source_samples: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(source_time_s) & np.isfinite(source_samples)
    if finite.sum() < 2:
        return np.full(len(query_time_s), np.nan)
    time_s = source_time_s[finite]
    samples = source_samples[finite]
    return np.interp(query_time_s, time_s, samples, left=np.nan, right=np.nan)


def zero_order_hold(
    source_time_s: np.ndarray,
    source_samples: np.ndarray,
    query_time_s: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(source_time_s) & np.isfinite(source_samples)
    if finite.sum() < 1:
        return np.full(len(query_time_s), np.nan)
    time_s = source_time_s[finite]
    samples = source_samples[finite]
    indices = np.searchsorted(time_s, query_time_s, side="right") - 1
    valid = (indices >= 0) & (query_time_s <= time_s[-1])
    output = np.full(len(query_time_s), np.nan)
    output[valid] = samples[indices[valid]]
    return output


def sampling_clock(
    decoded: MDF,
    *,
    clock_channel: str,
    start_s: float | None,
    finish_s: float | None,
    frequency_hz: float | None = None,
) -> np.ndarray:
    clock_time_s, _ = signal_arrays(decoded, clock_channel)
    finite_clock_s = clock_time_s[np.isfinite(clock_time_s)]
    if len(finite_clock_s) < 2:
        raise ValueError("The selected clock has fewer than two finite samples")
    effective_start_s = float(finite_clock_s[0]) if start_s is None else start_s
    effective_finish_s = float(finite_clock_s[-1]) if finish_s is None else finish_s
    if (
        not np.isfinite(effective_start_s)
        or not np.isfinite(effective_finish_s)
        or effective_finish_s <= effective_start_s
    ):
        raise ValueError("The conversion window must have finite start < finish")
    if frequency_hz is not None:
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        timestep_s = 1.0 / frequency_hz
        return np.arange(
            effective_start_s,
            effective_finish_s + timestep_s * 0.25,
            timestep_s,
        )
    selected = clock_time_s[
        np.isfinite(clock_time_s)
        & (clock_time_s >= effective_start_s)
        & (clock_time_s <= effective_finish_s)
    ]
    if len(selected) < 2:
        raise ValueError("Fewer than two clock samples are inside the time window")
    return selected


def sample_signals(
    decoded: MDF,
    specs: tuple[SignalSpec, ...],
    *,
    clock_channel: str,
    clock_output_name: str,
    start_s: float | None = None,
    finish_s: float | None = None,
    frequency_hz: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Sample only the requested signals inside the requested time window."""

    clock_s = sampling_clock(
        decoded,
        clock_channel=clock_channel,
        start_s=start_s,
        finish_s=finish_s,
        frequency_hz=frequency_hz,
    )
    clock_source_time_s, clock_source_samples = signal_arrays(decoded, clock_channel)
    columns: dict[str, np.ndarray] = {
        "time_s": clock_s - clock_s[0],
        "mf4_time_s": clock_s,
        clock_output_name: interpolate(
            clock_source_time_s, clock_source_samples, clock_s
        ),
    }
    metadata: dict[str, dict[str, object]] = {
        clock_output_name: {
            "source": clock_channel,
            "sampling": "clock",
            "native_median_rate_hz": source_rate_hz(clock_source_time_s),
        }
    }
    for spec in specs:
        source_name = resolve_channel(decoded, spec)
        if source_name is None:
            continue
        source_time_s, source_samples = signal_arrays(decoded, source_name)
        sampler = zero_order_hold if spec.sampling == "hold" else interpolate
        # A positive correction moves a delayed source backward: the value
        # logged at clock + shift is assigned to the row at clock.
        query_time_s = clock_s + spec.backward_time_shift_s
        columns[spec.output_name] = sampler(source_time_s, source_samples, query_time_s)
        metadata[spec.output_name] = {
            "source": source_name,
            "sampling": spec.sampling,
            "native_median_rate_hz": source_rate_hz(source_time_s),
            "backward_time_shift_s": spec.backward_time_shift_s,
            "time_shift_basis": (
                "HVC pack power versus motor torque * motor speed alignment"
                if spec.output_name == "battery_power_kw"
                and spec.backward_time_shift_s > 0
                else None
            ),
        }
    return columns, metadata


def write_csv(columns: dict[str, np.ndarray], output: Path) -> Path:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("All CSV columns must have equal length")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(columns)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names), strict=True))
    return output.resolve()


def parse_signal(raw: str, sampling: Literal["linear", "hold"]) -> SignalSpec:
    if "=" not in raw:
        raise ValueError(f"Signal must use OUTPUT_NAME=DECODED_CHANNEL: {raw}")
    output_name, channel_name = raw.split("=", maxsplit=1)
    if not output_name or not channel_name:
        raise ValueError(f"Signal must use OUTPUT_NAME=DECODED_CHANNEL: {raw}")
    return SignalSpec(output_name, (channel_name,), sampling=sampling)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mf4", nargs="?", type=Path, default=DEFAULT_MF4)
    parser.add_argument("--dbc-dir", type=Path, default=DEFAULT_DBC_DIR)
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--finish-s", type=float, default=None)
    parser.add_argument("--clock-channel", default=DEFAULT_CLOCK_CHANNEL)
    parser.add_argument("--clock-output-name", default="gps_speed_mps")
    parser.add_argument(
        "--frequency-hz",
        type=float,
        default=None,
        help="Uniform output rate. Omit to retain native clock timestamps.",
    )
    parser.add_argument("--signal", action="append", default=[])
    parser.add_argument("--hold-signal", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_specs = tuple(
        [parse_signal(raw, "linear") for raw in args.signal]
        + [parse_signal(raw, "hold") for raw in args.hold_signal]
    )
    specs = requested_specs or DEFAULT_SELECTED_SIGNALS
    decoded = decode_mf4(args.mf4, args.dbc_dir)
    columns, channel_metadata = sample_signals(
        decoded,
        specs,
        clock_channel=args.clock_channel,
        clock_output_name=args.clock_output_name,
        start_s=args.start_s,
        finish_s=args.finish_s,
        frequency_hz=args.frequency_hz,
    )
    output = write_csv(columns, args.output)
    clock_s = columns["mf4_time_s"]
    intervals_s = np.diff(clock_s)
    metadata = {
        "input_mf4": str(args.mf4.resolve()),
        "output_csv": str(output),
        "requested_start_mf4_time_s": args.start_s,
        "requested_finish_mf4_time_s": args.finish_s,
        "clock_channel": args.clock_channel,
        "requested_frequency_hz": args.frequency_hz,
        "samples": len(clock_s),
        "median_output_rate_hz": float(1.0 / np.median(intervals_s)),
        "selected_signal_count": len(channel_metadata),
        "channels": channel_metadata,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"csv: {output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
