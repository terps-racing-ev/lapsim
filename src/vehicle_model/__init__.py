"""Composable vehicle models used by the lap solvers and control replay.

Implementations live in electrical, aero, powertrain, and mech subpackages;
the coordinator remains in vehicle.py. The classes exported here are the
simple baseline implementations. The protocols describe the interfaces a
higher-fidelity replacement must expose.
"""

from .aero import Aero, AeroForces
from .electrical import Battery, Inverter, OCVPackBattery, RCTheveninBattery
from .interfaces import (
    AeroModel,
    BatteryModel,
    BrakeModel,
    ChassisModel,
    ChainDriveModel,
    ComponentModel,
    DrivetrainModel,
    FinalDriveModel,
    InverterModel,
    MotorModel,
    SuspensionModel,
    TireModel,
)
from .mech import (
    Brakes,
    Chassis,
    Pacejka61LateralModel,
    Suspension,
    Tire,
    TireNormalLoads,
    WheelSlip,
)
from .powertrain import ChainDrive, Drivetrain, FinalDrive, Motor
from .vehicle import Vehicle

__all__ = [
    "Aero",
    "AeroForces",
    "AeroModel",
    "Battery",
    "BatteryModel",
    "OCVPackBattery",
    "RCTheveninBattery",
    "BrakeModel",
    "Brakes",
    "Chassis",
    "ChassisModel",
    "ChainDrive",
    "ChainDriveModel",
    "ComponentModel",
    "Drivetrain",
    "DrivetrainModel",
    "FinalDrive",
    "FinalDriveModel",
    "Inverter",
    "InverterModel",
    "Motor",
    "MotorModel",
    "Pacejka61LateralModel",
    "Suspension",
    "SuspensionModel",
    "Tire",
    "TireModel",
    "TireNormalLoads",
    "Vehicle",
    "WheelSlip",
]
