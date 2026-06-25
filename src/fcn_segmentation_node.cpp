// C++ header
#include <memory>

// ROS header
#include <rclcpp/executors/events_cbg_executor/events_cbg_executor.hpp>

// local header
#include "fcn_segmentation/fcn_segmentation.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Create the node
  auto node = std::make_shared<fcn_segmentation::FCNSegmentation>();

  // EventsCBGExecutor: uses 10-15% less CPU than MultiThreadedExecutor,
  // supports multiple ROS time sources, and manages threading internally.
  rclcpp::executors::EventsCBGExecutor executor;

  // Add node to executor
  executor.add_node(node);

  RCLCPP_INFO(node->get_logger(), "Starting FCOS Object Detection with EventsCBGExecutor");

  // Spin with EventsCBGExecutor
  executor.spin();

  rclcpp::shutdown();

  return 0;
}
