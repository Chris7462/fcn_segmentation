#pragma once

// C++ header
#include <mutex>
#include <queue>

// Torch header
#include <torch/script.h>
#include <torch/torch.h>

// ROS header
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace fcn_segmentation
{

class FCNSegmentation : public rclcpp::Node
{
public:
  FCNSegmentation();
  ~FCNSegmentation() = default;

private:
  void img_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  void timer_callback();
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr seg_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::queue<sensor_msgs::msg::Image::SharedPtr> img_buff_;

  std::mutex mtx_;

  // Load the scripted model
  torch::jit::script::Module module_;
  torch::Device device_;
};

} // namespace fcn_segmentation
