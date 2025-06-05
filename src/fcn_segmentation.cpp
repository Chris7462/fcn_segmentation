// local header
#include "fcn_segmentation/fcn_segmentation.hpp"

namespace fcn_segmentation
{

FCNSegmentation::FCNSegmentation()
: Node("fcn_segmentation_node")
{
  rclcpp::QoS qos(10);
  img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    "kitti/camera/color/left/image_raw", qos, std::bind(
      &FCNSegmentation::img_callback, this, std::placeholders::_1));
}

void FCNSegmentation::img_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  img_buff_.push(msg);
}

} // namespace fcn_segmentation
