"""Composable vehicle models used by the lap solvers and control replay.

Implementations live in electrical, aero, powertrain, and mech subpackages;
the coordinator remains in vehicle.py. The classes exported here are the
simple baseline implementations. The protocols describe the interfaces a
higher-fidelity replacement must expose.
"""

from .aero import ActiveAero, ActiveAeroMode, Aero, AeroForces
from .electrical import Battery, Inverter, OCVPackBattery, RCTheveninBattery
from .interfaces import (
    AeroModel,
    BatteryModel,
    BrakeModel,
    ChassisModel,
    ChainDriveModel,
    ComponentModel,
    DrivetrainModel,
    InverterModel,
    MotorModel,
    SuspensionModel,
    TireModel,
    WheelAlignmentModel,
)
from .mech import (
    Brakes,
    CamberGainCurve,
    Chassis,
    Pacejka52UpcR20LateralModel,
    Pacejka52UpcR20LongitudinalModel,
    Pacejka61LateralModel,
    Suspension,
    Tire,
    TireCornerValues,
    TireForces,
    TireKinematics,
    TireNormalLoads,
    TireState,
    TireStates,
    WheelAlignment,
    WheelAlignmentCorner,
    WheelAlignmentState,
    WheelKinematics,
)
from .powertrain import ChainDrive, Drivetrain, Motor, MotorEfficiencyMap
from .vehicle import Vehicle

__all__ = [
    "Aero",
    "AeroForces",
    "ActiveAero",
    "ActiveAeroMode",
    "AeroModel",
    "Battery",
    "BatteryModel",
    "OCVPackBattery",
    "RCTheveninBattery",
    "BrakeModel",
    "Brakes",
    "CamberGainCurve",
    "Chassis",
    "ChassisModel",
    "ChainDrive",
    "ChainDriveModel",
    "ComponentModel",
    "Drivetrain",
    "DrivetrainModel",
    "Inverter",
    "InverterModel",
    "Motor",
    "MotorEfficiencyMap",
    "MotorModel",
    "Pacejka52UpcR20LateralModel",
    "Pacejka52UpcR20LongitudinalModel",
    "Pacejka61LateralModel",
    "Suspension",
    "SuspensionModel",
    "Tire",
    "TireCornerValues",
    "TireForces",
    "TireKinematics",
    "TireModel",
    "WheelAlignmentModel",
    "TireNormalLoads",
    "TireState",
    "TireStates",
    "Vehicle",
    "WheelAlignment",
    "WheelAlignmentCorner",
    "WheelAlignmentState",
    "WheelKinematics",
]
