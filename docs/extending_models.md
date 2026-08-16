# Extending component models

All baseline components are mutable dataclasses. Change a single parameter at
its owner, then validate before running:

```python
from vehicle_model import Vehicle

vehicle = Vehicle()
vehicle.drivetrain.motor.peak_power_w = 60_000.0
vehicle.drivetrain.inverter.efficiency = 0.96
vehicle.drivetrain.chain_drive.ratio = 4.1
vehicle.suspension = MySuspension(...)
vehicle.validate()
```

This nested structure also gives a future JSON loader or parameter sweeper an
unambiguous path such as `drivetrain.motor.peak_power_w`.

## Replacing a model

The protocols in `vehicle_model.interfaces` are structural: a replacement does
not have to inherit from the baseline class. It only needs to expose the
documented attributes and methods. Passing it to `Vehicle` or `Drivetrain`
performs lifecycle validation immediately.

The root package keeps common imports concise, while domain-qualified imports
make subteam ownership explicit:

```python
from vehicle_model import Vehicle
from vehicle_model.electrical import RCTheveninBattery
from vehicle_model.powertrain import Drivetrain, Motor
from vehicle_model.aero import Aero
from vehicle_model.mech import Suspension, Tire
```

Put a new implementation beside the baseline owned by the same subteam. Keep
cross-subteam coordination in `vehicle.py` or `powertrain/drivetrain.py`, not
in a solver.

For an incremental model, subclassing the baseline is convenient. This example
adds a motor temperature and thermal torque derating while retaining the
existing torque curve and power conversion:

```python
from dataclasses import dataclass, field

from vehicle_model import Drivetrain, Motor, Vehicle


@dataclass
class ThermalMotor(Motor):
    heat_capacity_j_per_k: float = 8_000.0
    cooling_w_per_k: float = 20.0
    ambient_temperature_c: float = 25.0
    derate_temperature_c: float = 90.0
    temperature_c: float = field(init=False, default=25.0)

    def validate(self) -> None:
        super().validate()
        if self.heat_capacity_j_per_k <= 0:
            raise ValueError("heat_capacity_j_per_k must be positive")

    def reset_state(self) -> None:
        super().reset_state()
        self.temperature_c = self.ambient_temperature_c

    def torque_limit_nm(self, motor_speed_rpm: float) -> float:
        cold_limit_nm = super().torque_limit_nm(motor_speed_rpm)
        if self.temperature_c <= self.derate_temperature_c:
            return cold_limit_nm
        return 0.75 * cold_limit_nm

    def update_state(
        self,
        motor_torque_nm: float,
        motor_speed_rpm: float,
        timestep_s: float,
    ) -> None:
        super().update_state(motor_torque_nm, motor_speed_rpm, timestep_s)
        mechanical_power_w = (
            motor_torque_nm
            * motor_speed_rpm
            * 2.0
            * 3.14159
            / 60.0
        )
        loss_w = mechanical_power_w * (1.0 / self.efficiency - 1.0)
        cooling_w = self.cooling_w_per_k * max(
            self.temperature_c - self.ambient_temperature_c,
            0.0,
        )
        self.temperature_c += (
            (loss_w - cooling_w) * timestep_s / self.heat_capacity_j_per_k
        )

    def update_telemetry(self, telemetry: dict[str, float]) -> None:
        super().update_telemetry(telemetry)
        telemetry["motor.temperature_c"] = self.temperature_c


vehicle = Vehicle(drivetrain=Drivetrain(motor=ThermalMotor()))
```

For a fundamentally different implementation, implement the corresponding
protocol directly. Useful contracts are:

| Protocol | Used by |
|---|---|
| `MotorModel` | `Drivetrain` torque and power calculations |
| `InverterModel` | `Drivetrain` DC/electrical conversions |
| `ChainDriveModel` | wheel/motor torque and power conversions |
| `BatteryModel` | pack-terminal power limiting and state update |
| `TireModel` | acceleration, cornering, and braking limits |
| `SuspensionModel` | four tire normal loads |
| `AeroModel` | drag and axle downforce |
| `DrivetrainModel` | `Vehicle` and both spatial solvers |

Inspect `vehicle_model/interfaces.py` for exact signatures and the applicable
subteam package for baseline validation conventions.

## Rules for new components

1. Store parameters and mutable state in the component that physically owns
   them.
2. Implement `validate()` and fail early for impossible parameter sets.
3. Implement `reset_state()` so repeated simulations are independent.
4. Keep limit queries free of unrelated side effects.
5. Apply component dynamic state changes in `update_state(..., timestep_s)`;
   `Vehicle` supplies the internal timestep derived from its spatial cell.
6. Write current scalar signals in `update_telemetry(...)` using a stable
   component namespace.
7. Return shared result dataclasses at component boundaries.
8. Add a focused unit test for the model and one integration test through
   `Vehicle` or a solver.

Validation runs when the component is constructed and once at the beginning
of each solver or replay. It is intentionally not repeated every timestep, so
validation of a large lookup table does not dominate a long simulation. After
mutating parameters manually, call `vehicle.validate()` before a direct
`Vehicle.update_state()` loop.

If new physics needs data absent from a protocol, add an explicit operating
point or result dataclass rather than passing an unstructured dictionary.
If motor, inverter, and battery behavior becomes inseparably coupled, replace
the `DrivetrainModel` as a unit instead of hiding cross-component logic in a
solver.

## Parameter sweeps

Create a fresh vehicle for each sample so thermal state, SOC, and other future
state cannot leak between cases:

```python
from vehicle_model import Vehicle

for ratio in (3.3, 3.455, 3.6, 3.8):
    vehicle = Vehicle()
    vehicle.drivetrain.chain_drive.ratio = ratio
    vehicle.validate()
    # Run a solver and store the result here.
```

Compatibility aliases such as `drivetrain.final_drive_ratio` remain available
for older scripts. New code should use `drivetrain.chain_drive.ratio`.
