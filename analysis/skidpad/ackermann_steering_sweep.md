# 9.125 m skidpad Ackermann steering sweep

Source: `logs/skidpad_testing.mf4`, started 2026-09-16 23:56:16 EDT. The
steering channel is `CAN2.DBF_Steering_Angle.DBF_Steering_Angle_Deg`; GNSS speed
and IMU yaw rate are from CAN1. Every steering sample has its validity flag set.
The driver-identified 9.125 m intervals are **320–355 s left** and **419–435 s
right**, measured from the start of the log. The IMU yaw sign is opposite the
vehicle model's turn sign in these runs; the driver labels take precedence.

The model uses a 9.125 m center path radius, default vehicle parameters, 1 degree
**front toe-in per wheel**, and zero rear toe. `anti_ackermann_fraction` is 0 for
parallel steer and 1 for full anti-Ackermann. At each measured median speed,
`_solve_controls_at_speed` solves the steady longitudinal and lateral balance
for each sweep setting. Positive model steering means a left turn. The logged
steering angle is positive in the identified left run.

Within the identified intervals, samples were kept when GNSS speed exceeded
3 m/s, yaw direction matched the run, and `abs(abs(speed / yaw_rate) - 9.125)`
was below 1 m. A +0.25 deg/s stationary offset was applied to the IMU yaw rate
for this radius filter. The sensor angle is the raw median and is **not**
converted to model road-wheel angle.

| Turn | Time (s) | Samples | Speed (m/s) | Measured radius (m) | Sensor median (deg) | Sensor IQR (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Left | 320–355 | 145 | 6.136 | 9.048 | +42.42 | +41.61 to +43.69 |
| Right | 419–435 | 78 | 6.526 | 8.916 | -33.49 | -34.62 to -32.40 |

Model steering control in degrees at those measured speeds:

| Anti-Ackermann fraction | Left | Right |
| ---: | ---: | ---: |
| 0.00 | +9.788 | -9.758 |
| 0.25 | +9.789 | -9.758 |
| 0.50 | +9.792 | -9.759 |
| 0.75 | +9.795 | -9.762 |
| 1.00 | +9.800 | -9.766 |

The largest change across the entire sweep is 0.012 degrees of model steering
control. A steering sensor zero and a sensor-to-road-wheel conversion have not
been established, so an absolute angle error cannot be computed. With only two
turn medians, an unknown sensor zero and scale can be fitted to both at *every*
Ackermann setting. Even with those calibrations, this center-steering signal is
weak evidence for Ackermann: the setup mainly changes the *difference between
front wheel angles*. At a 9.76-degree left model control, the front right minus
front left angle changes from 2.00 degrees at fraction 0 to 3.30 degrees at
fraction 1, including the specified toe.

**Conclusion:** The available steering telemetry does not distinguish an
Ackermann setting. A direct left and right front road-wheel angle measurement at
known steering-wheel positions would resolve the wheel-angle split; a measured
steering sensor zero and ratio are also needed to compare the log to the model.
