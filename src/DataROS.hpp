#ifndef DATAROS_HPP
#define DATAROS_HPP

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <ros/ros.h>
#include "message_filters/cache.h"
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>

#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>

class DataROS
{
public:
  DataROS(int argc, char **argv);
  ~DataROS();

  void subscribe(void (*getData)(const Eigen::MatrixXd points, const Eigen::MatrixXd colors));
  void poseCallback(const std_msgs::StringConstPtr& string_msg);
  void depthCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
private:

  bool operational;
  message_filters::Cache<sensor_msgs::PointCloud2> cache;
  void (*subscriber)(const Eigen::MatrixXd points, const Eigen::MatrixXd colors);
  
  Eigen::Matrix4d rOffset;
  void pointsFromMsg(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, Eigen::Matrix4d& cam);
  Eigen::Affine3f findPlaneRotation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
};


#endif // DATAROS_HPP
