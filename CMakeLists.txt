cmake_minimum_required(VERSION 3.8)
project(pi_ros2_interface)

# Good defaults
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(quadrotor_msgs REQUIRED)

# -------------------------
# C library: pi_protocol
# -------------------------
add_library(pi_protocol STATIC
  src/pi-protocol.c
  src/pi-messages.c
)

# IMPORTANT: Make headers visible both in-build and after install
target_include_directories(pi_protocol PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# If your C code needs math functions (sinf), link libm
target_link_libraries(pi_protocol m)

# -------------------------
# ROS2 node
# -------------------------
add_executable(pi_bridge_node
  src/pi_bridge_node.cpp
)

target_link_libraries(pi_bridge_node
  pi_protocol
)

ament_target_dependencies(pi_bridge_node
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  Eigen3
  tf2_ros
  quadrotor_msgs
)

target_include_directories(pi_bridge_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# -------------------------
# Install rules (THIS is what makes ros2 find the package)
# -------------------------
install(TARGETS
  pi_protocol
  pi_bridge_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# Export include dirs + library for downstream packages
ament_export_include_directories(include)
ament_export_libraries(pi_protocol)
ament_package()
