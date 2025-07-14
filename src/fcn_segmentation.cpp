// C++ header
#include <string>
#include <chrono>

// OpenCV header
#include <opencv2/highgui.hpp>

// ROS header
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>

// local header
#include "fcn_segmentation/fcn_segmentation.hpp"


namespace fcn_segmentation
{

FCNSegmentation::FCNSegmentation()
: Node("fcn_segmentation_node"),
  processing_image_(false),
  image_sequence_number_(0),
  last_processed_sequence_(0)
{
  // Initialize ROS2 parameters with validation
  if (!initialize_parameters()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize parameters");
    rclcpp::shutdown();
    return;
  }

  // Initialize TensorRT inferencer
  if (!initialize_inferencer()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize TensorRT inferencer");
    rclcpp::shutdown();
    return;
  }

  // Initialize ROS2 components
  initialize_ros_components();

  RCLCPP_INFO(get_logger(), "FCN Segmentation node initialized successfully");
}

FCNSegmentation::~FCNSegmentation()
{
  RCLCPP_INFO(get_logger(), "FCN Segmentation node shutting down");
}

bool FCNSegmentation::initialize_parameters()
{
  try {
    // ROS2 parameters
    input_topic_ = declare_parameter("input_topic", std::string("kitti/camera/color/left/image_raw"));
    output_topic_ = declare_parameter("output_topic", std::string("fcn_segmentation"));
    output_overlay_topic_ = declare_parameter("output_topic_overlay", std::string("fcn_segmentation_overlay"));
    queue_size_ = declare_parameter<int>("queue_size", 10);
    processing_frequency_ = declare_parameter<double>("processing_frequency", 40.0);

    // Declare and get parameters with validation
    engine_filename_ = declare_parameter("engine_file", std::string("fcn_resnet50_1238x374.trt"));
    config_.width = declare_parameter<int>("width", 1238);
    config_.height = declare_parameter<int>("height", 374);
    config_.num_classes = declare_parameter<int>("num_classes", 21);
    config_.warmup_iterations = declare_parameter<int>("warmup_iterations", 2);
    config_.log_level = static_cast<tensorrt_inferencer::Logger::Severity>(
      declare_parameter<int>("log_level", 3)); // Set log level

    // Validation
    if (engine_filename_.empty()) {
      RCLCPP_ERROR(get_logger(), "Engine filename cannot be empty");
      return false;
    }

    if (config_.width <= 0 || config_.height <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid image dimensions: %dx%d", config_.width, config_.height);
      return false;
    }

    if (config_.num_classes <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid number of classes: %d", config_.num_classes);
      return false;
    }

    if (processing_frequency_ <= 0) {
      RCLCPP_ERROR(get_logger(), "Invalid processing frequency: %.2f Hz", processing_frequency_);
      return false;
    }

    // Construct engine file path
    fs::path package_path = ament_index_cpp::get_package_share_directory("fcn_segmentation");
    engine_path_ = package_path / "engines" / engine_filename_;

    RCLCPP_INFO(get_logger(), "Parameters initialized - Engine: %s, Dims: %dx%d, Classes: %d",
      engine_path_.c_str(), config_.width, config_.height, config_.num_classes);

    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception during parameter initialization: %s", e.what());
    return false;
  }
}

bool FCNSegmentation::initialize_inferencer()
{
  // Check if engine file exists
  if (!fs::exists(engine_path_)) {
    RCLCPP_ERROR(get_logger(), "Engine file does not exist: %s", engine_path_.c_str());
    return false;
  }

  try {
    inferencer_ = std::make_shared<tensorrt_inferencer::TensorRTInferencer>(engine_path_, config_);

    if (!inferencer_) {
      RCLCPP_ERROR(get_logger(), "Failed to create TensorRTInferencer instance");
      return false;
    }

    RCLCPP_INFO(get_logger(), "TensorRT inferencer initialized successfully");
    return true;

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception creating TensorRTInferencer: %s", e.what());
    return false;
  }
}

void FCNSegmentation::initialize_ros_components()
{
  // Configure QoS profile for reliable image transport
  rclcpp::QoS image_qos(queue_size_);
  image_qos.reliability(rclcpp::ReliabilityPolicy::BestEffort);
  image_qos.durability(rclcpp::DurabilityPolicy::Volatile);
  image_qos.history(rclcpp::HistoryPolicy::KeepLast);

  // Create subscriber with proper callback binding
  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_topic_, image_qos,
    std::bind(&FCNSegmentation::image_callback, this, std::placeholders::_1)
  );

  // Create publisher
  fcn_pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, image_qos);
  fcn_overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(output_overlay_topic_ , image_qos);

  // Create timer for processing at specified frequency
  auto timer_period = std::chrono::duration<double>(1.0 / processing_frequency_);
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
    std::bind(&FCNSegmentation::timer_callback, this)
  );

  RCLCPP_INFO(get_logger(), "ROS components initialized - Input: %s, Output: %s, Frequency: %.1f Hz",
              input_topic_.c_str(), output_topic_.c_str(), processing_frequency_);
}

void FCNSegmentation::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Thread-safe image storage
  std::lock_guard<std::mutex> lock(image_mutex_);

  try {
    // Convert ROS image to OpenCV format
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    if (!cv_ptr || cv_ptr->image.empty()) {
      RCLCPP_WARN(get_logger(), "Received empty or invalid image");
      return;
    }

    // Store the latest image and header
    latest_image_ = cv_ptr->image.clone();
    latest_header_ = msg->header;
    image_sequence_number_.fetch_add(1); // Increment sequence number for new image

  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception in image callback: %s", e.what());
  }
}

void FCNSegmentation::timer_callback()
{
  // Skip processing if already processing or no image received
  if (processing_image_.load()) {
    return;
  }

  // Check if we have subscribers before processing
  if (fcn_pub_->get_subscription_count() == 0) {
    return;
  }

  // Check if we have a new image to process
  uint64_t current_sequence = image_sequence_number_.load();
  if (current_sequence == last_processed_sequence_) {
    return; // No new image to process
  }

  cv::Mat current_image;
  std_msgs::msg::Header current_header;

  // Thread-safe image retrieval
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (latest_image_.empty()) {
      return;
    }
    current_image = latest_image_.clone();
    current_header = latest_header_;
  }

  // Set processing flag
  processing_image_.store(true);

  try {
    // Process the image
    cv::Mat segmentation_result = process_image(current_image);
    cv::Mat overlay = inferencer_->create_overlay(current_image, segmentation_result, 0.5f);

    if (!segmentation_result.empty()) {
      publish_segmentation_result(segmentation_result, current_header);
      publish_overlay_result(overlay, current_header);
      last_processed_sequence_ = current_sequence; // Update only on successful processing
    } else {
      RCLCPP_WARN(get_logger(), "Segmentation processing returned empty result");
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception during image processing: %s", e.what());
  }

  // Clear processing flag
  processing_image_.store(false);
}

cv::Mat FCNSegmentation::process_image(const cv::Mat & input_image)
{
  if (input_image.empty()) {
    RCLCPP_WARN(get_logger(), "Input image is empty");
    return cv::Mat();
  }

  try {
    // Resize image to model input size if necessary
    cv::Mat processed_image;
    if (input_image.cols != config_.width || input_image.rows != config_.height) {
      cv::resize(input_image, processed_image, cv::Size(config_.width, config_.height), 0, 0, cv::INTER_LINEAR);
    } else {
      processed_image = input_image;
    }

    // Run inference
    auto segmentation = inferencer_->infer(processed_image);

    if (segmentation.empty()) {
      RCLCPP_WARN(get_logger(), "Inference returned empty result");
      return cv::Mat();
    }

    // Resize back to original size if necessary
    if (segmentation.cols != input_image.cols || segmentation.rows != input_image.rows) {
      cv::Mat resized_mask;
      cv::resize(segmentation, resized_mask,
        cv::Size(input_image.cols, input_image.rows), 0, 0, cv::INTER_NEAREST);
      return resized_mask;
    }

    return segmentation;

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Exception during image processing: %s", e.what());
    return cv::Mat();
  }
}

void FCNSegmentation::publish_segmentation_result(const cv::Mat & segmentation,
  const std_msgs::msg::Header & header)
{
  try {
    // Convert OpenCV image back to ROS message
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8; // Always colored segmentation
    cv_image.image = segmentation;

    // Publish the result
    auto output_msg = cv_image.toImageMsg();
    fcn_pub_->publish(*output_msg);

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception during result publishing: %s", e.what());
  }
}

void FCNSegmentation::publish_overlay_result(const cv::Mat & overlay,
  const std_msgs::msg::Header & header)
{
  try {
    // Convert OpenCV image back to ROS message
    cv_bridge::CvImage cv_image;
    cv_image.header = header;
    cv_image.encoding = sensor_msgs::image_encodings::BGR8; // Always colored segmentation
    cv_image.image = overlay;

    // Publish the result
    auto output_msg = cv_image.toImageMsg();
    fcn_overlay_pub_->publish(*output_msg);

  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Exception during result publishing: %s", e.what());
  }
}

} // namespace fcn_segmentation
