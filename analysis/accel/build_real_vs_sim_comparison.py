"""Build aligned real-vs-sim acceleration data for launch 13.

The replay case applies the recorded inverter motor-torque command and measured
front/rear brake pressures as functions of elapsed launch time.  The limit case
requests the vehicle model's full 230 N m motor torque while retaining all
modeled tire, motor, inverter, battery, and drivetrain constraints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lapsim import Controls  # noqa: E402
from vehicle_model import RCTheveninBattery, Vehicle  # noqa: E402


LAUNCH_NUMBER = 13
COURSE_DISTANCE_M = 75.0
MAXIMUM_TORQUE_REQUEST_NM = 230.0
HVC_LAG_S = 0.09
GNSS_LAG_S = 0.3072


def _find_launch_window(data: pd.DataFrame, launch_number: int) -> tuple[int, int]:
    active = data["VCU_Launch_Active"].fillna(0.0).to_numpy() > 0.5
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    if launch_number < 1 or launch_number > len(starts):
        raise ValueError(f"launch {launch_number} is unavailable; found {len(starts)}")
    return int(starts[launch_number - 1]), int(ends[launch_number - 1])


def _interp(time_s: np.ndarray, values: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    finite = np.isfinite(time_s) & np.isfinite(values)
    return np.interp(query_s, time_s[finite], values[finite])


def _integrate_speed(time_s: np.ndarray, speed_mps: np.ndarray) -> np.ndarray:
    result = np.zeros_like(time_s)
    result[1:] = np.cumsum(
        0.5 * (speed_mps[1:] + speed_mps[:-1]) * np.diff(time_s)
    )
    return result


def _crossing_time(time_s: np.ndarray, distance_m: np.ndarray) -> float | None:
    indices = np.flatnonzero(distance_m >= COURSE_DISTANCE_M)
    if not len(indices):
        return None
    upper = int(indices[0])
    if upper == 0:
        return float(time_s[0])
    lower = upper - 1
    fraction = (COURSE_DISTANCE_M - distance_m[lower]) / (
        distance_m[upper] - distance_m[lower]
    )
    return float(time_s[lower] + fraction * (time_s[upper] - time_s[lower]))


def _simulation_row(vehicle: Vehicle, requested_torque_nm: float, front_psi: float, rear_psi: float) -> dict[str, float]:
    telemetry = vehicle.telemetry_snapshot()
    return {
        "elapsed_time_s": vehicle.time_s,
        "distance_m": vehicle.distance_m,
        "speed_mps": vehicle.speed_mps,
        "accel_mps2": vehicle.longitudinal_acceleration_mps2,
        "motor_torque_request_nm": requested_torque_nm,
        "motor_torque_delivered_nm": telemetry["motor.torque_nm"],
        "motor_speed_rpm": telemetry["motor.speed_rpm"],
        "motor_mechanical_power_kw": telemetry["motor.mechanical_power_w"] / 1000.0,
        "pack_power_kw": telemetry["battery.power_w"] / 1000.0,
        "pack_current_a": telemetry["battery.current_a"],
        "pack_voltage_v": telemetry["battery.terminal_voltage_v"],
        "front_brake_pressure_psi": front_psi,
        "rear_brake_pressure_psi": rear_psi,
        "drive_force_n": telemetry["vehicle.drive_force_n"],
        "traction_limited": telemetry["limits.traction_active"],
        "motor_envelope_limited": telemetry["limits.motor_envelope_active"],
    }


def _simulate(
    initial_soc: float,
    controls_at_time,
    *,
    distance_step_m: float = 0.01,
) -> pd.DataFrame:
    vehicle = Vehicle(
        battery=RCTheveninBattery(initial_state_of_charge=initial_soc),
    )
    vehicle.validate()
    vehicle.reset_state()
    requested, front_psi, rear_psi = controls_at_time(0.0)
    rows = [_simulation_row(vehicle, requested, front_psi, rear_psi)]
    while vehicle.distance_m < COURSE_DISTANCE_M:
        requested, front_psi, rear_psi = controls_at_time(vehicle.time_s)
        controls = Controls(
            motor_torque_request_nm=max(0.0, requested),
            front_brake_pressure_psi=max(0.0, front_psi),
            rear_brake_pressure_psi=max(0.0, rear_psi),
        )
        step_m = min(distance_step_m, COURSE_DISTANCE_M - vehicle.distance_m)
        vehicle.update_state(controls, step_m)
        rows.append(_simulation_row(vehicle, requested, front_psi, rear_psi))
        if vehicle.time_s > 10.0:
            raise RuntimeError("simulation did not finish within 10 seconds")
    return pd.DataFrame(rows)


def _resample_simulation(data: pd.DataFrame, time_grid_s: np.ndarray) -> pd.DataFrame:
    result = {"elapsed_time_s": time_grid_s}
    source_time = data["elapsed_time_s"].to_numpy(float)
    for column in data.columns:
        if column == "elapsed_time_s":
            continue
        values = data[column].to_numpy(float)
        aligned = np.interp(time_grid_s, source_time, values)
        aligned[time_grid_s > source_time[-1]] = np.nan
        result[column] = aligned
    return pd.DataFrame(result)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(predicted)
    return float(np.sqrt(np.mean((predicted[finite] - actual[finite]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    launch_start, launch_end = _find_launch_window(data, LAUNCH_NUMBER)
    launch_start_s = float(data.loc[launch_start, "timestamp_s"])
    launch_end_s = float(data.loc[launch_end, "timestamp_s"])
    raw_time_s = data["timestamp_s"].to_numpy(float)
    initial_soc = float(data.loc[launch_start, "HVC_SOC_Percent"] / 100.0)

    input_time_s = raw_time_s - launch_start_s
    torque_command_nm = data["INV_Commanded_Torque"].to_numpy(float)
    front_pressure_psi = data["VCU_BSE_PSI"].to_numpy(float)
    rear_pressure_psi = data["VCU_RB_Rear_BSE_PSI"].to_numpy(float)

    def replay_controls(elapsed_s: float) -> tuple[float, float, float]:
        query = np.asarray([elapsed_s])
        return (
            float(_interp(input_time_s, torque_command_nm, query)[0]),
            float(_interp(input_time_s, front_pressure_psi, query)[0]),
            float(_interp(input_time_s, rear_pressure_psi, query)[0]),
        )

    def limit_controls(elapsed_s: float) -> tuple[float, float, float]:
        del elapsed_s
        return MAXIMUM_TORQUE_REQUEST_NM, 0.0, 0.0

    replay_raw = _simulate(initial_soc, replay_controls)
    limit_raw = _simulate(initial_soc, limit_controls)
    maximum_time_s = max(
        5.5,
        float(replay_raw["elapsed_time_s"].iloc[-1]),
        float(limit_raw["elapsed_time_s"].iloc[-1]),
    )
    time_grid_s = np.round(np.arange(0.0, maximum_time_s + 0.025, 0.05), 10)
    replay = _resample_simulation(replay_raw, time_grid_s)
    limit = _resample_simulation(limit_raw, time_grid_s)

    query_abs_s = launch_start_s + time_grid_s
    vcu_speed_mps = _interp(
        raw_time_s, data["VCU_Speed_MPH"].to_numpy(float) * 0.44704, query_abs_s
    )
    gnss_speed_mps = _interp(
        raw_time_s,
        data["Speed"].to_numpy(float),
        query_abs_s + GNSS_LAG_S,
    )
    imu_raw = _interp(raw_time_s, data["AccelerationX"].to_numpy(float), query_abs_s)
    prelaunch = data[
        (data["timestamp_s"] >= launch_start_s - 0.5)
        & (data["timestamp_s"] < launch_start_s)
    ]["AccelerationX"].to_numpy(float)
    imu_bias = float(np.nanmedian(prelaunch))
    imu_corrected = imu_raw - imu_bias
    vcu_distance_m = _integrate_speed(time_grid_s, vcu_speed_mps)
    gnss_distance_m = _integrate_speed(time_grid_s, gnss_speed_mps)

    real = pd.DataFrame(
        {
            "elapsed_time_s": time_grid_s,
            "wheel_equiv_distance_m": vcu_distance_m,
            "gnss_distance_m": gnss_distance_m,
            "vcu_wheel_equiv_speed_mps": vcu_speed_mps,
            "gnss_speed_mps": gnss_speed_mps,
            "imu_accel_raw_mps2": imu_raw,
            "imu_accel_baseline_corrected_mps2": imu_corrected,
            "apps_percent": _interp(raw_time_s, data["VCU_APPS_Value"].to_numpy(float), query_abs_s),
            "vcu_torque_request_nm": _interp(raw_time_s, data["VCU_INV_Torque_Cmd"].to_numpy(float), query_abs_s),
            "inverter_torque_command_nm": _interp(raw_time_s, torque_command_nm, query_abs_s),
            "inverter_torque_feedback_nm": _interp(raw_time_s, data["INV_Torque_Feedback"].to_numpy(float), query_abs_s),
            "motor_speed_rpm": _interp(raw_time_s, data["INV_Motor_Speed"].to_numpy(float), query_abs_s),
            "pack_power_kw": _interp(raw_time_s, data["HVC_Pack_Power_kW"].to_numpy(float), query_abs_s + HVC_LAG_S),
            "pack_current_a": _interp(raw_time_s, data["HVC_Pack_Current_A"].to_numpy(float), query_abs_s + HVC_LAG_S),
            "pack_voltage_v": _interp(raw_time_s, data["HVC_Batt_Voltage_V"].to_numpy(float), query_abs_s + HVC_LAG_S),
            "front_brake_pressure_psi": _interp(raw_time_s, front_pressure_psi, query_abs_s),
            "rear_brake_pressure_psi": _interp(raw_time_s, rear_pressure_psi, query_abs_s),
        }
    )

    real.to_csv(args.output_dir / "real_launch_13_aligned.csv", index=False)
    replay.to_csv(args.output_dir / "sim_replayed_controls.csv", index=False)
    limit.to_csv(args.output_dir / "sim_absolute_limit.csv", index=False)

    real_crossing_s = _crossing_time(time_grid_s, vcu_distance_m)
    gnss_crossing_s = _crossing_time(time_grid_s, gnss_distance_m)
    replay_time_s = float(replay_raw["elapsed_time_s"].iloc[-1])
    limit_time_s = float(limit_raw["elapsed_time_s"].iloc[-1])
    within_replay = time_grid_s <= replay_time_s
    summary = {
        "source_file": str(args.input.resolve()),
        "launch_number": LAUNCH_NUMBER,
        "launch_start_timestamp_s": launch_start_s,
        "launch_end_timestamp_s": launch_end_s,
        "initial_soc_percent": 100.0 * initial_soc,
        "course_distance_m": COURSE_DISTANCE_M,
        "real_wheel_equivalent_75m_time_s": real_crossing_s,
        "real_gnss_integrated_75m_time_s": gnss_crossing_s,
        "sim_replay_75m_time_s": replay_time_s,
        "sim_limit_75m_time_s": limit_time_s,
        "replay_minus_real_wheel_equiv_time_s": None if real_crossing_s is None else replay_time_s - real_crossing_s,
        "limit_minus_replay_time_s": limit_time_s - replay_time_s,
        "real_peak_wheel_equiv_speed_mps": float(np.nanmax(vcu_speed_mps)),
        "real_peak_gnss_speed_mps": float(np.nanmax(gnss_speed_mps)),
        "replay_finish_speed_mps": float(replay_raw["speed_mps"].iloc[-1]),
        "limit_finish_speed_mps": float(limit_raw["speed_mps"].iloc[-1]),
        "real_peak_corrected_accel_mps2": float(np.nanmax(imu_corrected)),
        "replay_peak_accel_mps2": float(np.nanmax(replay_raw["accel_mps2"])),
        "limit_peak_accel_mps2": float(np.nanmax(limit_raw["accel_mps2"])),
        "real_peak_pack_power_kw": float(np.nanmax(real["pack_power_kw"])),
        "replay_peak_pack_power_kw": float(np.nanmax(replay_raw["pack_power_kw"])),
        "limit_peak_pack_power_kw": float(np.nanmax(limit_raw["pack_power_kw"])),
        "replay_rmse_vs_real": {
            "wheel_equiv_speed_mps": _rmse(vcu_speed_mps[within_replay], replay.loc[within_replay, "speed_mps"].to_numpy(float)),
            "gnss_speed_mps": _rmse(gnss_speed_mps[within_replay], replay.loc[within_replay, "speed_mps"].to_numpy(float)),
            "baseline_corrected_accel_mps2": _rmse(imu_corrected[within_replay], replay.loc[within_replay, "accel_mps2"].to_numpy(float)),
            "motor_speed_rpm": _rmse(real.loc[within_replay, "motor_speed_rpm"].to_numpy(float), replay.loc[within_replay, "motor_speed_rpm"].to_numpy(float)),
            "motor_torque_feedback_nm": _rmse(real.loc[within_replay, "inverter_torque_feedback_nm"].to_numpy(float), replay.loc[within_replay, "motor_torque_delivered_nm"].to_numpy(float)),
            "pack_power_kw": _rmse(real.loc[within_replay, "pack_power_kw"].to_numpy(float), replay.loc[within_replay, "pack_power_kw"].to_numpy(float)),
            "pack_current_a": _rmse(real.loc[within_replay, "pack_current_a"].to_numpy(float), replay.loc[within_replay, "pack_current_a"].to_numpy(float)),
            "pack_voltage_v": _rmse(real.loc[within_replay, "pack_voltage_v"].to_numpy(float), replay.loc[within_replay, "pack_voltage_v"].to_numpy(float)),
        },
        "alignment_and_assumptions": {
            "time_zero": "first VCU_Launch_Active sample for launch 13",
            "replay_motor_control": "INV_Commanded_Torque, the actual post-launch-schedule and post-power-limit inverter command",
            "replay_brake_controls": "VCU_BSE_PSI front and VCU_RB_Rear_BSE_PSI rear",
            "limit_control": "constant 230 N m motor request with all modeled constraints active",
            "limit_initial_soc": "same as launch 13 for an apples-to-apples control comparison",
            "gnss_lag_correction_s": GNSS_LAG_S,
            "hvc_lag_correction_s": HVC_LAG_S,
            "imu_correction": f"subtracted {imu_bias:.3f} m/s^2 median AccelerationX over the 0.5 s pre-launch window",
            "real_distance_warning": "VCU speed is rear-wheel/motor-derived and is not independent vehicle speed; GNSS is shown separately",
            "model_warning": "current point-mass simulator kinematically locks motor speed to vehicle speed and cannot reproduce the logged wheel-speed surge/recovery",
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
