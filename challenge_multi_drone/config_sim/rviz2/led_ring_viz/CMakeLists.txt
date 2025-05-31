cmake_minimum_required(VERSION 3.5)
project(led_ring_visualizer)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Add the executable
add_executable(led_ring_visualizer src/led_ring_viz.cpp)
ament_target_dependencies(led_ring_visualizer rclcpp std_msgs visualization_msgs tf2_ros geometry_msgs tf2_geometry_msgs)

# Install the executable
install(TARGETS
  led_ring_visualizer
  DESTINATION lib/${PROJECT_NAME})

# Install launch files (if any)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
