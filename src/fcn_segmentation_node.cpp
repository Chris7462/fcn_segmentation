#include "fcn_segmentation/fcn_segmentation.hpp"


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fcn_segmentation::FCNSegmentation>());
  rclcpp::shutdown();
  return 0;
}
