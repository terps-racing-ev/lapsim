"""Shared wheel ordering and signs, independent of any subsystem model."""

WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
# Left-side tires have +1, right-side tires -1 for tire-local transformations.
WHEEL_SIDE_SIGNS = (1.0, -1.0, 1.0, -1.0)
