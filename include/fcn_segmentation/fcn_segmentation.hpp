#pragma once

// C++ header
#include <queue>
#include <mutex>
#include <memory>

// ROS header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// Local header
#include "tensorrt_inferencer/tensorrt_inferencer.hpp"


namespace fcn_segmentation
{

class FCNSegmentation : public rclcpp::Node
{
public:
  FCNSegmentation();
  ~FCNSegmentation() = default;

private:
  void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void timer_callback();

private:
  std::shared_ptr<TensorRTInferencer> inferencer_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr fcn_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::queue<sensor_msgs::msg::Image::SharedPtr> img_buff_;
  std::mutex mtx_;
};

} // namespace fcn_segmentation
