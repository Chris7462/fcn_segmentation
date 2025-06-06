// C++ header
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

// OpenCV header
#include <cv_bridge/cv_bridge.hpp>

// local header
#include "fcn_segmentation/fcn_segmentation.hpp"


namespace fcn_segmentation
{

namespace fs = std::filesystem;
using namespace std::chrono_literals;

FCNSegmentation::FCNSegmentation()
: Node("fcn_segmentation_node"),
  device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
  fs::path model_path = declare_parameter("model_path", fs::path());
  fs::path model_file = model_path / declare_parameter("model_file", std::string());

  if (!fs::exists(model_file)) {
    RCLCPP_ERROR(get_logger(), "Load model failed");
    rclcpp::shutdown();
  }

  rclcpp::QoS qos(10);
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "kitti/camera/color/left/image_raw", qos, std::bind(
      &FCNSegmentation::img_callback, this, std::placeholders::_1));

  seg_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
    "fcn_segmentation", qos);

  timer_ = this->create_wall_timer(
    25ms, std::bind(&FCNSegmentation::timer_callback, this));

  RCLCPP_INFO(get_logger(), "Using device: %s", device_.is_cuda() ? "CUDA" : "CPU");
  module_ = torch::jit::load(model_file.string());
  module_.to(device_);
  module_.eval();
}

void FCNSegmentation::img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  img_buff_.push(msg);
}

void FCNSegmentation::timer_callback()
{
  sensor_msgs::msg::Image::SharedPtr input_msg;
  rclcpp::Time current_time = this->now();
  RCLCPP_INFO_STREAM(get_logger(), "Current time = " << current_time.nanoseconds());
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!img_buff_.empty()) {
      auto msg = img_buff_.front();
      RCLCPP_INFO_STREAM(get_logger(), "Header time = " << rclcpp::Time(msg->header.stamp).nanoseconds());
      RCLCPP_INFO_STREAM(get_logger(), "Time diff = " << (current_time - rclcpp::Time(msg->header.stamp)).seconds());

      if ((current_time - rclcpp::Time(msg->header.stamp)).seconds() <= 0.1) {
        input_msg = msg;
        img_buff_.pop();
      } else {
        RCLCPP_WARN(get_logger(), "Timestamp unaligned, skipping old frame.");
        img_buff_.pop();  // discard old frame
      }
    }
  }

  // No image to process
  if (!input_msg) {
    return;
  }

  try {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat image = cv_bridge::toCvCopy(input_msg, "bgr8")->image;
    // Convert to float and normalize to [0, 1]
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // Convert OpenCV Mat (H x W x C) to torch::Tensor (1 x C x H x W)
    torch::Tensor input_tensor = torch::from_blob(
        image.data,
        {1, image.rows, image.cols, 3},
        torch::kFloat32).permute({0, 3, 1, 2}).clone().to(device_);  // clone() to own the data

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    //// Normalize per channel
    //const std::vector<double> mean = {0.485, 0.456, 0.406};
    //const std::vector<double> std = {0.229, 0.224, 0.225};
    //for (int c = 0; c < 3; ++c) {
    //  input_tensor[0][c] = input_tensor[0][c].sub(mean[c]).div(std[c]);
    //}

    // Forward pass
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs{input_tensor};
    //at::Tensor output = module_.forward(inputs).toTensor();
    // TODO: This should be update. In Python just save the ['out'] result, instead of Dict
    auto output_dict = module_.forward(inputs).toGenericDict();
    torch::Tensor output = output_dict.at("out").toTensor();

    RCLCPP_INFO(get_logger(), "Inference time: %.3f ms", duration.count() * 1000);

    // (optional) move back to CPU for post-processing or visualization
    //output = output.to(torch::kCPU);

    // Convert OpenCV image to ROS Image message
    auto out_msg = input_msg;
    //auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();
    out_msg->header.frame_id = "cam2_link";
    out_msg->header.stamp = current_time;
    seg_pub_->publish(*out_msg);

  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "CV_Bridge exception: %s", e.what());
  }
}

} // namespace fcn_segmentation
