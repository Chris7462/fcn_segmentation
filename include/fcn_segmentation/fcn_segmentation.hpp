#pragma once

// C++ header
#include <mutex>
#include <queue>

// Torch header
#include <torch/torch.h>
#include <torch/script.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

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
  torch::Device device_;
  cv::cuda::HostMem pinned_host_;
  cv::cuda::GpuMat gpu_bgr_;
  cv::cuda::GpuMat gpu_float_;
  torch::jit::script::Module module_;
};

} // namespace fcn_segmentation
