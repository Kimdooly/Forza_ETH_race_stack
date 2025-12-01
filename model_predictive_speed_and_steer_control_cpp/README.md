# model_predictive_speed_and_steer_control_cpp

A ROS 2 (rclcpp) port of the kinematic MPC speed and steer demo. The node publishes the same race stack topics as the Python version so it can drop into existing launch files.

## Build

```bash
colcon build --symlink-install --packages-select model_predictive_speed_and_steer_control_cpp
source install/setup.bash
```

## Run

Launch the node with the bundled demo parameters:

```bash
ros2 launch model_predictive_speed_and_steer_control_cpp model_predictive_speed_and_steer_control.launch.py
```

Parameters (loop_dt, horizon_length, target_speed, path geometry, and initial state) can be edited in `config/demo_params.yaml` and will be picked up automatically when using `--symlink-install`.
