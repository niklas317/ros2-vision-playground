#ifndef WAYPOINT_CONTROLLER__WAYPOINT_CONTROL_NODE_HPP_
#define WAYPOINT_CONTROLLER__WAYPOINT_CONTROL_NODE_HPP_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"

namespace waypoint_controller
{

class WaypointControlNode : public rclcpp::Node
{
public:
  WaypointControlNode();

private:
  using Waypoint = std::pair<double, double>;

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);

  std::size_t find_closest_waypoint(
    double x,
    double y) const;

  std::size_t find_lookahead_waypoint(
    std::size_t closest_index,
    double x,
    double y) const;

  double calculate_steering_angle(
    double vehicle_x,
    double vehicle_y,
    double vehicle_yaw,
    const Waypoint & target) const;

  double quaternion_to_yaw(
    double x,
    double y,
    double z,
    double w) const;

  double normalize_angle(double angle) const;

  // Track waypoints
  std::vector<Waypoint> waypoints_;

  // Controller parameters
  double lookahead_distance_;
  double wheelbase_;
  double max_steering_angle_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_pub_;
};

}  // namespace waypoint_controller

#endif  // WAYPOINT_CONTROLLER__WAYPOINT_CONTROL_NODE_HPP_