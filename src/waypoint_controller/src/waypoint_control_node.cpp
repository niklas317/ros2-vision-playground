#include "waypoint_controller/waypoint_control_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace waypoint_controller
{

WaypointControlNode::WaypointControlNode()
: Node("waypoint_control_node")
{
  // Pure-Pursuit/controller parameters
  lookahead_distance_ =
    declare_parameter<double>("lookahead_distance", 0.5);

  wheelbase_ =
    declare_parameter<double>("wheelbase", 0.22);

  max_steering_angle_ =
    declare_parameter<double>("max_steering_angle", 0.6108652);

  // Waypoints are provided as:
  // [x0, y0, x1, y1, x2, y2, ...]
  const auto waypoint_values =
    declare_parameter<std::vector<double>>(
      "waypoints",
      std::vector<double>{});

  if (waypoint_values.size() < 4 ||
      waypoint_values.size() % 2 != 0)
  {
    throw std::runtime_error(
      "Parameter 'waypoints' must contain at least two x/y pairs.");
  }

  for (std::size_t i = 0; i < waypoint_values.size(); i += 2) {
    waypoints_.emplace_back(
      waypoint_values[i],
      waypoint_values[i + 1]);
  }

  odom_sub_ =
    create_subscription<nav_msgs::msg::Odometry>(
      "/odom",
      10,
      std::bind(
        &WaypointControlNode::odom_callback,
        this,
        std::placeholders::_1));

  steering_pub_ =
    create_publisher<std_msgs::msg::Float64>(
      "/steering_angle",
      10);

  RCLCPP_INFO(
    get_logger(),
    "Waypoint controller started with %zu waypoints.",
    waypoints_.size());
}


void WaypointControlNode::odom_callback(
  const nav_msgs::msg::Odometry::SharedPtr msg)
{
  const double vehicle_x =
    msg->pose.pose.position.x;

  const double vehicle_y =
    msg->pose.pose.position.y;

  const auto & q =
    msg->pose.pose.orientation;

  const double vehicle_yaw =
    quaternion_to_yaw(
      q.x,
      q.y,
      q.z,
      q.w);

  // Find the current position on the track
  const std::size_t closest_index =
    find_closest_waypoint(
      vehicle_x,
      vehicle_y);

  // Select a point sufficiently far ahead
  const std::size_t target_index =
    find_lookahead_waypoint(
      closest_index,
      vehicle_x,
      vehicle_y);

  const Waypoint & target =
    waypoints_[target_index];

  const double steering_angle =
    calculate_steering_angle(
      vehicle_x,
      vehicle_y,
      vehicle_yaw,
      target);

  std_msgs::msg::Float64 steering_msg;
  steering_msg.data = steering_angle;

  steering_pub_->publish(steering_msg);
}


std::size_t WaypointControlNode::find_closest_waypoint(
  double x,
  double y) const
{
  std::size_t closest_index = 0;
  double smallest_distance =
    std::numeric_limits<double>::max();

  for (std::size_t i = 0; i < waypoints_.size(); ++i) {
    const double dx =
      waypoints_[i].first - x;

    const double dy =
      waypoints_[i].second - y;

    const double distance_squared =
      dx * dx + dy * dy;

    if (distance_squared < smallest_distance) {
      smallest_distance = distance_squared;
      closest_index = i;
    }
  }

  return closest_index;
}


std::size_t WaypointControlNode::find_lookahead_waypoint(
  std::size_t closest_index,
  double x,
  double y) const
{
  // Search forward along the closed waypoint loop until
  // the waypoint is at least lookahead_distance away.
  for (std::size_t offset = 1;
       offset <= waypoints_.size();
       ++offset)
  {
    const std::size_t index =
      (closest_index + offset) %
      waypoints_.size();

    const double dx =
      waypoints_[index].first - x;

    const double dy =
      waypoints_[index].second - y;

    const double distance =
      std::hypot(dx, dy);

    if (distance >= lookahead_distance_) {
      return index;
    }
  }

  return closest_index;
}


double WaypointControlNode::calculate_steering_angle(
  double vehicle_x,
  double vehicle_y,
  double vehicle_yaw,
  const Waypoint & target) const
{
  const double dx =
    target.first - vehicle_x;

  const double dy =
    target.second - vehicle_y;

  const double target_distance =
    std::hypot(dx, dy);

  if (target_distance < 1e-6) {
    return 0.0;
  }

  // Heading from vehicle to target waypoint
  const double target_heading =
    std::atan2(dy, dx);

  // Relative target angle in vehicle coordinates
  const double alpha =
    normalize_angle(
      target_heading - vehicle_yaw);

  // Standard Pure Pursuit steering law
  double steering_angle =
    std::atan2(
      2.0 * wheelbase_ * std::sin(alpha),
      target_distance);

  steering_angle =
    std::clamp(
      steering_angle,
      -max_steering_angle_,
      max_steering_angle_);

  return steering_angle;
}


double WaypointControlNode::quaternion_to_yaw(
  double x,
  double y,
  double z,
  double w) const
{
  // Extract yaw from quaternion
  const double siny_cosp =
    2.0 * (w * z + x * y);

  const double cosy_cosp =
    1.0 - 2.0 * (y * y + z * z);

  return std::atan2(
    siny_cosp,
    cosy_cosp);
}


double WaypointControlNode::normalize_angle(
  double angle) const
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }

  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }

  return angle;
}

}  // namespace waypoint_controller


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<
      waypoint_controller::WaypointControlNode>());

  rclcpp::shutdown();

  return 0;
}