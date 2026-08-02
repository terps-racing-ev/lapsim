"""Unit conversion constants and functions."""

from math import degrees, pi, radians


METERS_PER_INCH = 0.0254
METERS_PER_FOOT = 0.3048
METERS_PER_MILE = 1_609.344
KILOGRAMS_PER_POUND = 0.45359237
NEWTONS_PER_POUND_FORCE = 4.4482216152605
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3_600.0
RADIANS_PER_REVOLUTION = 2.0 * pi


def pounds_to_kilograms(mass_lb: float) -> float:
    """Convert avoirdupois pounds to kilograms."""

    return mass_lb * KILOGRAMS_PER_POUND


def pounds_force_to_newtons(force_lbf: float) -> float:
    """Convert pounds-force to newtons."""

    return force_lbf * NEWTONS_PER_POUND_FORCE


def inches_to_meters(distance_in: float) -> float:
    """Convert inches to metres."""

    return distance_in * METERS_PER_INCH


def feet_to_meters(distance_ft: float) -> float:
    """Convert feet to metres."""

    return distance_ft * METERS_PER_FOOT


def miles_per_hour_to_meters_per_second(speed_mph: float) -> float:
    """Convert miles per hour to metres per second."""

    meters_per_hour = speed_mph * METERS_PER_MILE
    return meters_per_hour / SECONDS_PER_HOUR


def meters_per_second_to_miles_per_hour(speed_mps: float) -> float:
    """Convert metres per second to miles per hour."""

    meters_per_hour = speed_mps * SECONDS_PER_HOUR
    return meters_per_hour / METERS_PER_MILE


def revolutions_per_minute_to_radians_per_second(speed_rpm: float) -> float:
    """Convert rotational speed from RPM to radians per second."""

    revolutions_per_second = speed_rpm / SECONDS_PER_MINUTE
    return revolutions_per_second * RADIANS_PER_REVOLUTION


def radians_per_second_to_revolutions_per_minute(speed_rad_s: float) -> float:
    """Convert rotational speed from radians per second to RPM."""

    revolutions_per_second = speed_rad_s / RADIANS_PER_REVOLUTION
    return revolutions_per_second * SECONDS_PER_MINUTE


def degrees_to_radians(angle_deg: float) -> float:
    """Convert degrees to radians."""

    return radians(angle_deg)


def radians_to_degrees(angle_rad: float) -> float:
    """Convert radians to degrees."""

    return degrees(angle_rad)
