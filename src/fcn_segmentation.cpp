// C++ header
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

// ROS2 header
#include <cv_bridge/cv_bridge.hpp>

#include "c10/cuda/CUDAStream.h"    // for at::cuda::CUDAStream
#include "c10/cuda/CUDAGuard.h"     // for at::cuda::CUDAStreamGuard

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
  int width = declare_parameter<int>("width", 1238);
  int height = declare_parameter<int>("height", 374);

  if (!fs::exists(model_file)) {
    RCLCPP_ERROR(get_logger(), "Load model failed");
    rclcpp::shutdown();
  }

  // Pinned host memory
  pinned_host_ = cv::cuda::HostMem{
    cv::Size(width, height), CV_8UC3,
    cv::cuda::HostMem::SHARED
  };

  // Setting up the model
  module_ = torch::jit::load(model_file.string(), device_);
  module_.eval();

  rclcpp::QoS qos(10);
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "kitti/camera/color/left/image_raw", qos, std::bind(
      &FCNSegmentation::img_callback, this, std::placeholders::_1));

  seg_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
    "fcn_segmentation", qos);

  timer_ = this->create_wall_timer(
    25ms, std::bind(&FCNSegmentation::timer_callback, this));

  RCLCPP_INFO(get_logger(), "Using device: %s", device_.is_cuda() ? "CUDA" : "CPU");
}

void FCNSegmentation::img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  img_buff_.push(msg);
}

void FCNSegmentation::timer_callback()
{
  auto callback_start = std::chrono::high_resolution_clock::now();

  rclcpp::Time current_time = this->now();
  sensor_msgs::msg::Image::SharedPtr input_msg;
  {
    std::lock_guard<std::mutex> lock(mtx_);
    RCLCPP_INFO_STREAM(get_logger(), "Buffer size = " << img_buff_.size());
    if (img_buff_.empty()) {
      return;
    }

    auto msg = img_buff_.front();
    img_buff_.pop();

    RCLCPP_INFO_STREAM(get_logger(), "Current Time = " << current_time.nanoseconds());
    RCLCPP_INFO_STREAM(get_logger(), "Message Time = " << msg->header.stamp.sec << msg->header.stamp.nanosec);
    RCLCPP_INFO_STREAM(get_logger(), "Time diff = " << (current_time - rclcpp::Time(msg->header.stamp)).seconds() << "s");

    // drop frames older than 0.1s
    if ((current_time - rclcpp::Time(msg->header.stamp)).seconds() > 0.1) { // 100ms
      RCLCPP_WARN(get_logger(), "Timestamp unaligned, skipping old frame.");
      return;
    }
    input_msg = msg;
  }

  // 2) ROS image → cv::Mat (on CPU)
  auto cv_start = std::chrono::high_resolution_clock::now();
  cv::Mat cpu_bgr;
  try {
    cpu_bgr = cv_bridge::toCvShare(input_msg, "bgr8")->image;
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
    return;
  }
  auto cv_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cv_duration = cv_end - cv_start;
  RCLCPP_INFO(get_logger(), "CV time: %.3f ms", cv_duration.count() * 1000);

  // 3) copy into page‐locked (pinned) host memory
  auto cpu_gpu_start = std::chrono::high_resolution_clock::now();
  size_t byte_count = cpu_bgr.total() * cpu_bgr.elemSize();
  std::memcpy(pinned_host_.data, cpu_bgr.data, byte_count);
  auto cpu_gpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_gpu_duration = cpu_gpu_end - cpu_gpu_start;
  RCLCPP_INFO(get_logger(), "CPU to GPU time: %.3f ms", cpu_gpu_duration.count() * 1000);


  // 4) wrap into a GpuMat header and preprocess
  auto gpu_norm_start = std::chrono::high_resolution_clock::now();
  cv::cuda::GpuMat gpu_bgr = pinned_host_.createGpuMatHeader();
  cv::cuda::GpuMat gpu_float;
  gpu_bgr.convertTo(gpu_float, CV_32FC3, 1.0f/255.0f);       // normalize to [0,1]
  cv::cuda::GpuMat gpu_rgb;
  cv::cuda::cvtColor(gpu_float, gpu_rgb, cv::COLOR_BGR2RGB); // model expects RGB
  auto gpu_norm_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_norm_duration = gpu_norm_end - gpu_norm_start;
  RCLCPP_INFO(get_logger(), "GPU norm time: %.3f ms", gpu_norm_duration.count() * 1000);


  // 5) wrap the GpuMat data as a CUDA tensor (no intermediate host copy)
  //    note: we guard the correct CUDA stream so torch & OpenCV stay in sync
  auto tensor_start = std::chrono::high_resolution_clock::now();
  auto cuda_stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::CUDAStreamGuard guard(cuda_stream);
  auto tensor = torch::from_blob(
    gpu_rgb.ptr<float>(),
    {1, gpu_rgb.rows, gpu_rgb.cols, 3},
    torch::kFloat32);
  tensor = tensor.permute({0,3,1,2})               // NHWC → NCHW
                    .to(device_, /*non_blocking=*/true);
  auto tensor_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tensor_duration = tensor_end - tensor_start;
  RCLCPP_INFO(get_logger(), "Tensor time: %.3f ms", tensor_duration.count() * 1000);

  // 6) run inference
  auto inf_start = std::chrono::high_resolution_clock::now();
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs{tensor};
  //at::Tensor output = module_.forward(inputs).toTensor();
  // TODO: This should be update. In Python just save the ['out'] result, instead of Dict
  auto output_dict = module_.forward(inputs).toGenericDict();
  torch::Tensor output = output_dict.at("out").toTensor();
  auto inf_end = std::chrono::high_resolution_clock::now();
  auto inf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inf_end - inf_start);
  RCLCPP_INFO(get_logger(), "Inference time: %ld ms", inf_duration.count());

//seg_pub_->publish(*input_msg);


    //auto cv_start = std::chrono::high_resolution_clock::now();
    //std::memcpy(pinned_host_.data, input_msg->data.data(), input_msg->data.size());
    //gpu_bgr_ = pinned_host_.createGpuMatHeader();
    //gpu_bgr_.convertTo(gpu_float_, CV_32FC3, 1.0F/255.0F);
    //auto cv_end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> cv_duration = cv_end - cv_start;
    //RCLCPP_INFO(get_logger(), "CV time: %.3f ms", cv_duration.count() * 1000);

    //auto tensor_start = std::chrono::high_resolution_clock::now();
    //auto input_tensor = at::from_blob(
    //  gpu_float_.data,
    //  {1, gpu_float_.rows, gpu_float_.cols, 3},
    //  torch::TensorOptions{}.dtype(torch::kFloat32).device(device_)
    //).permute({0, 3, 1, 2});
    //auto tensor_end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> tensor_duration = tensor_end - tensor_start;
    //RCLCPP_INFO(get_logger(), "Tensor time: %.3f ms", tensor_duration.count() * 1000);

    //auto start = std::chrono::high_resolution_clock::now();
    //torch::NoGradGuard no_grad;
    //std::vector<torch::jit::IValue> inputs{input_tensor};
    ////at::Tensor output = module_.forward(inputs).toTensor();
    //// TODO: This should be update. In Python just save the ['out'] result, instead of Dict
    //auto output_dict = module_.forward(inputs).toGenericDict();
    //torch::Tensor output = output_dict.at("out").toTensor();

    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> duration = end - start;
    //RCLCPP_INFO(get_logger(), "Inference time: %.3f ms", duration.count() * 1000);

    //try {
    //  auto cv_start = std::chrono::high_resolution_clock::now();

    //  cv::Mat image = cv_bridge::toCvShare(input_msg, "bgr8")->image;
    //  // Convert to float and normalize to [0, 1]
    //  image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    //  auto cv_end = std::chrono::high_resolution_clock::now();
    //  std::chrono::duration<double> cv_duration = cv_end - cv_start;
    //  RCLCPP_INFO(get_logger(), "CV time: %.3f ms", cv_duration.count() * 1000);

    //  auto tensor_start = std::chrono::high_resolution_clock::now();
    //  // Convert OpenCV Mat (H x W x C) to torch::Tensor (1 x C x H x W)
    //  torch::Tensor input_tensor = torch::from_blob(
    //      image.data,
    //      {1, image.rows, image.cols, 3},
    //      //torch::kFloat32).permute({0, 3, 1, 2}).clone().to(device_);  // clone() to own the data
    //      torch::kFloat32).permute({0, 3, 1, 2}).to(device_);  // clone() to own the data

    //  auto tensor_end = std::chrono::high_resolution_clock::now();
    //  std::chrono::duration<double> tensor_duration = tensor_end - tensor_start;
    //  RCLCPP_INFO(get_logger(), "Tensor time: %.3f ms", tensor_duration.count() * 1000);

    //  //// Normalize per channel
    //  //const std::vector<double> mean = {0.485, 0.456, 0.406};
    //  //const std::vector<double> std = {0.229, 0.224, 0.225};
    //  //for (int c = 0; c < 3; ++c) {
    //  //  input_tensor[0][c] = input_tensor[0][c].sub(mean[c]).div(std[c]);
    //  //}

    //  // Forward pass
    //  auto start = std::chrono::high_resolution_clock::now();
    //  torch::NoGradGuard no_grad;
    //  std::vector<torch::jit::IValue> inputs{input_tensor};
    //  //at::Tensor output = module_.forward(inputs).toTensor();
    //  // TODO: This should be update. In Python just save the ['out'] result, instead of Dict
    //  auto output_dict = module_.forward(inputs).toGenericDict();
    //  torch::Tensor output = output_dict.at("out").toTensor();

    //  auto end = std::chrono::high_resolution_clock::now();
    //  std::chrono::duration<double> duration = end - start;
    //  RCLCPP_INFO(get_logger(), "Inference time: %.3f ms", duration.count() * 1000);

    //  // (optional) move back to CPU for post-processing or visualization
    //  //output = output.to(torch::kCPU);

    //  // Convert OpenCV image to ROS Image message
    //  auto out_msg = input_msg;
    //  //auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", cv_image).toImageMsg();
    //  out_msg->header.frame_id = "cam2_link";
    //  out_msg->header.stamp = current_time;
    //  seg_pub_->publish(*out_msg);

    //} catch (cv_bridge::Exception & e) {
    //  RCLCPP_ERROR(get_logger(), "CV_Bridge exception: %s", e.what());
    //}

  auto callback_end = std::chrono::high_resolution_clock::now();
  auto callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end - callback_start);
  RCLCPP_INFO(get_logger(), "Callback time: %ld ms", callback_duration.count());

//seg_pub_->publish(*input_msg);
}

} // namespace fcn_segmentation
