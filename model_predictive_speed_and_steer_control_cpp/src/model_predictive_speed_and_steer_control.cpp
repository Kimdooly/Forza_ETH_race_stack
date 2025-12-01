#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr double PI = 3.14159265358979323846;
constexpr double WB = 2.5;  // wheel base
constexpr double HALF_WB_RATIO = 0.5;  // for beta term
constexpr double MAX_STEER = PI / 4.0;  // 45 deg
constexpr double MAX_ACCEL = 3.0;
constexpr double MIN_ACCEL = -4.0;
constexpr double MAX_SPEED = 15.0;

inline double wrapToPi(double angle)
{
  while (angle > PI) angle -= 2.0 * PI;
  while (angle < -PI) angle += 2.0 * PI;
  return angle;
}

struct State
{
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double v{0.0};
  double accel{0.0};
  double steer{0.0};
};

struct ReferencePath
{
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> yaw;
  std::vector<double> curvature;
};

struct MpcResult
{
  double accel{0.0};
  double steer{0.0};
  std::vector<State> predicted;
};

}  // namespace

class ModelPredictiveSpeedAndSteerControl : public rclcpp::Node
{
public:
  ModelPredictiveSpeedAndSteerControl()
  : Node("model_predictive_speed_and_steer_control"),
    loop_dt_(declare_parameter("loop_dt", 0.05)),
    horizon_length_(declare_parameter<int>("horizon_length", 10)),
    target_speed_(declare_parameter("target_speed", 2.5)),
    path_radius_(declare_parameter("path_radius", 12.0)),
    path_points_(declare_parameter<int>("path_points", 800))
  {
    state_.x = declare_parameter("initial_x", 0.0);
    state_.y = declare_parameter("initial_y", 0.0);
    state_.yaw = declare_parameter("initial_yaw", 0.0);
    state_.v = declare_parameter("initial_speed", 0.0);

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
    state_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/mpc_controller/states", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/mpc_controller/current_position", 10);
    predicted_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/mpc_controller/predicted_position", 10);

    buildReferencePath();

    timer_ = create_wall_timer(std::chrono::duration<double>(loop_dt_), std::bind(&ModelPredictiveSpeedAndSteerControl::onTimer, this));

    RCLCPP_INFO(get_logger(), "C++ MPC demo node initialized with %zu reference points and horizon %d", reference_.x.size(), horizon_length_);
  }

private:
  void onTimer()
  {
    const auto result = solveMpc();
    state_.accel = result.accel;
    state_.steer = result.steer;

    state_ = propagateState(state_, loop_dt_);

    publishDrive();
    publishStates();
    publishPose();
    publishPrediction(result.predicted);
  }

  void buildReferencePath()
  {
    reference_ = ReferencePath{};
    reference_.x.reserve(path_points_);
    reference_.y.reserve(path_points_);
    reference_.yaw.reserve(path_points_);
    reference_.curvature.reserve(path_points_);

    // Generate a closed circle track to match the Python demo behavior.
    for (int i = 0; i < path_points_; ++i)
    {
      const double theta = (2.0 * PI * static_cast<double>(i)) / static_cast<double>(path_points_);
      const double x = path_radius_ * std::cos(theta);
      const double y = path_radius_ * std::sin(theta);
      reference_.x.push_back(x);
      reference_.y.push_back(y);
      reference_.yaw.push_back(theta + PI / 2.0);
      reference_.curvature.push_back(1.0 / path_radius_);
    }
  }

  int nearestIndex(const State & state) const
  {
    int nearest = 0;
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < reference_.x.size(); ++i)
    {
      const double dx = state.x - reference_.x[i];
      const double dy = state.y - reference_.y[i];
      const double dist = dx * dx + dy * dy;
      if (dist < best)
      {
        best = dist;
        nearest = static_cast<int>(i);
      }
    }
    return nearest;
  }

  Eigen::Matrix<double, 4, 4> getA(double v, double yaw, double delta) const
  {
    Eigen::Matrix<double, 4, 4> A = Eigen::Matrix<double, 4, 4>::Identity();
    A(0, 2) = loop_dt_ * std::cos(yaw);
    A(0, 3) = -loop_dt_ * v * std::sin(yaw);
    A(1, 2) = loop_dt_ * std::sin(yaw);
    A(1, 3) = loop_dt_ * v * std::cos(yaw);
    A(3, 2) = loop_dt_ * std::tan(delta) / WB;
    return A;
  }

  Eigen::Matrix<double, 4, 2> getB(double v, double delta) const
  {
    Eigen::Matrix<double, 4, 2> B = Eigen::Matrix<double, 4, 2>::Zero();
    B(2, 0) = loop_dt_;
    B(3, 1) = loop_dt_ * v / (WB * std::pow(std::cos(delta), 2));
    return B;
  }

  State propagateState(State state, double dt) const
  {
    state.v = std::clamp(state.v + state.accel * dt, 0.0, MAX_SPEED);
    const double beta = std::atan(HALF_WB_RATIO * std::tan(state.steer));
    state.x += state.v * std::cos(state.yaw + beta) * dt;
    state.y += state.v * std::sin(state.yaw + beta) * dt;
    state.yaw = wrapToPi(state.yaw + (state.v / WB) * std::sin(beta) * dt);
    return state;
  }

  Eigen::Matrix4d qMatrix() const
  {
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;
    Q(2, 2) = 0.5;
    Q(3, 3) = 0.5;
    return Q;
  }

  Eigen::Matrix2d rMatrix() const
  {
    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = 0.05;
    R(1, 1) = 0.05;
    return R;
  }

  Eigen::Matrix2d rdMatrix() const
  {
    Eigen::Matrix2d Rd = Eigen::Matrix2d::Zero();
    Rd(0, 0) = 0.01;
    Rd(1, 1) = 0.5;
    return Rd;
  }

  MpcResult solveMpc()
  {
    const int nearest = nearestIndex(state_);
    const auto ref = getReferenceStates(nearest);

    const Eigen::Matrix4d Q = qMatrix();
    const Eigen::Matrix4d Qf = Q;
    const Eigen::Matrix2d R = rMatrix();
    const Eigen::Matrix2d Rd = rdMatrix();

    Eigen::Matrix<double, 4, 4> A;
    Eigen::Matrix<double, 4, 2> B;

    // Backward Riccati recursion for time-varying LQR.
    std::vector<Eigen::Matrix<double, 4, 4>> P(horizon_length_ + 1, Qf);
    std::vector<Eigen::Matrix<double, 2, 4>> K(horizon_length_);

    for (int i = horizon_length_ - 1; i >= 0; --i)
    {
      const double ref_v = ref[i + 1].v;
      const double ref_yaw = ref[i + 1].yaw;
      const double ref_delta = ref[i + 1].steer;

      A = getA(ref_v, ref_yaw, ref_delta);
      B = getB(ref_v, ref_delta);

      const Eigen::Matrix2d G = (R + B.transpose() * P[i + 1] * B);
      Eigen::Matrix2d G_inv = G.inverse();
      K[i] = G_inv * (B.transpose() * P[i + 1] * A);

      const Eigen::Matrix<double, 4, 4> identity = Eigen::Matrix<double, 4, 4>::Identity();
      Eigen::Matrix<double, 4, 4> Ad = identity - B * K[i];
      P[i] = A.transpose() * P[i + 1] * Ad + Q;
    }

    // Forward rollout with rate and magnitude limits and Rd penalization.
    std::vector<State> predicted;
    predicted.reserve(horizon_length_ + 1);
    predicted.push_back(state_);

    double prev_accel = state_.accel;
    double prev_steer = state_.steer;

    for (int i = 0; i < horizon_length_; ++i)
    {
      Eigen::Vector4d x_err;
      x_err << predicted.back().x - ref[i].x, predicted.back().y - ref[i].y, wrapToPi(predicted.back().yaw - ref[i].yaw), predicted.back().v - ref[i].v;

      Eigen::Vector2d u = -K[i] * x_err;
      // Rd term for smoothness.
      Eigen::Vector2d du;
      du << u[0] - prev_accel, u[1] - prev_steer;
      u -= Rd * du;

      double accel_cmd = std::clamp(u[0], MIN_ACCEL, MAX_ACCEL);
      double steer_cmd = std::clamp(u[1], -MAX_STEER, MAX_STEER);

      State next = predicted.back();
      next.accel = accel_cmd;
      next.steer = steer_cmd;
      next = propagateState(next, loop_dt_);

      predicted.push_back(next);
      prev_accel = accel_cmd;
      prev_steer = steer_cmd;
    }

    MpcResult result;
    result.accel = predicted[1].accel;
    result.steer = predicted[1].steer;
    result.predicted = predicted;
    return result;
  }

  std::vector<State> getReferenceStates(int nearest_index) const
  {
    std::vector<State> refs(horizon_length_ + 1);
    for (int i = 0; i <= horizon_length_; ++i)
    {
      const size_t idx = (nearest_index + i) % reference_.x.size();
      refs[i].x = reference_.x[idx];
      refs[i].y = reference_.y[idx];
      refs[i].yaw = reference_.yaw[idx];
      refs[i].v = target_speed_;
      refs[i].steer = std::clamp(reference_.curvature[idx] * WB, -MAX_STEER, MAX_STEER);
    }
    return refs;
  }

  void publishDrive()
  {
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = now();
    msg.drive.speed = state_.v;
    msg.drive.acceleration = state_.accel;
    msg.drive.steering_angle = state_.steer;
    drive_pub_->publish(msg);
  }

  void publishStates()
  {
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {state_.x, state_.y, state_.yaw, state_.v, state_.steer, state_.accel};
    state_pub_->publish(msg);
  }

  void publishPose()
  {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = now();
    msg.pose.position.x = state_.x;
    msg.pose.position.y = state_.y;
    msg.pose.orientation.z = std::sin(state_.yaw * 0.5);
    msg.pose.orientation.w = std::cos(state_.yaw * 0.5);
    pose_pub_->publish(msg);
  }

  void publishPrediction(const std::vector<State> & predicted)
  {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = now();
    marker.header.frame_id = "map";
    marker.ns = "mpc_prediction";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.scale.z = 0.25;
    marker.color.a = 0.9;
    marker.color.r = 0.0;
    marker.color.g = 0.5;
    marker.color.b = 1.0;

    for (const auto & s : predicted)
    {
      geometry_msgs::msg::Point p;
      p.x = s.x;
      p.y = s.y;
      p.z = 0.0;
      marker.points.push_back(p);
    }

    markers.markers.push_back(marker);
    predicted_pub_->publish(markers);
  }

  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr predicted_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  double loop_dt_;
  int horizon_length_;
  double target_speed_;
  double path_radius_;
  int path_points_;

  ReferencePath reference_{};
  State state_{};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ModelPredictiveSpeedAndSteerControl>());
  rclcpp::shutdown();
  return 0;
}

