cmake_minimum_required(VERSION 3.8)
project(waypoint_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(waypoint_control_node
  src/waypoint_control_node.cpp
)

ament_target_dependencies(waypoint_control_node
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
)

target_include_directories(waypoint_control_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS
  waypoint_control_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
