cmake_minimum_required(VERSION 3.20)

project(fcn_segmentation
  VERSION 1.0.0
  DESCRIPTION "FCN segmentation ROS2 node"
  LANGUAGES CXX)

# Default to a Release build if the user doesn't specify one
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(fcn_trt_backend REQUIRED)

add_executable(fcn_segmentation_node
  src/fcn_segmentation_node.cpp
  src/fcn_segmentation.cpp)

target_include_directories(fcn_segmentation_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(fcn_segmentation_node PRIVATE
  ${rclcpp_TARGETS}
  ${sensor_msgs_TARGETS}
  ${ament_index_cpp_TARGETS}
  ${cv_bridge_TARGETS}
  ${fcn_trt_backend_TARGETS})

target_compile_features(fcn_segmentation_node PRIVATE cxx_std_17)  # Require C++17

target_compile_options(fcn_segmentation_node PRIVATE
  -Wall
  -Wextra
  -Wpedantic)

install(TARGETS fcn_segmentation_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY param
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
