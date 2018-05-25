#include "DataROS.hpp"

#include "Eigen/Geometry"

#include <math.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <boost/foreach.hpp>

#include <pcl/PCLPointCloud2.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>


DataROS::DataROS(int argc, char **argv)
{
  rOffset = Eigen::Matrix4d::Identity();
  ros::init(argc, argv, "rtsr");
  ros::start();
}

void DataROS::subscribe(void (*getData)(const Eigen::MatrixXd points, const Eigen::MatrixXd colors)) {  
  DataROS::subscriber = getData;

  ros::NodeHandle n;
  ros::Rate loop_rate(1);

  ros::Subscriber depth_sub = n.subscribe("/camera/depth_registered/points", 1, &DataROS::depthCallback, this);
  ros::Subscriber pose_sub = n.subscribe("pose", 1, &DataROS::poseCallback, this);
  cache.setCacheSize(100);

  ros::spin();
}

void DataROS::poseCallback(const std_msgs::StringConstPtr& string_msg)
{
  std::stringstream ss;
  ss << string_msg->data.c_str();

  double timestamp;
  ss >> timestamp;

  double x, y, z, q0, q1, q2, q3;
  ss >> x >> y >> z >> q0 >> q1 >> q2 >> q3;

  ros::Time ti(timestamp);
  const sensor_msgs::PointCloud2ConstPtr& cloud_msg_1 = cache.getElemBeforeTime(ti);
  const sensor_msgs::PointCloud2ConstPtr& cloud_msg_2 = cache.getElemAfterTime(ti);
  if (cloud_msg_1 == NULL || cloud_msg_2 == NULL) return; // wait until we have both - check which is closer

  double d_1 = timestamp - cloud_msg_1->header.stamp.toSec();
  double d_2 = cloud_msg_2->header.stamp.toSec() - timestamp;

  const Eigen::Quaterniond quat(q3,q0,q1,q2);
  
  const Eigen::Vector3d vec(x, y, z);
  const Eigen::Translation3d trans = Eigen::Translation3d(vec);

  const Eigen::Affine3d t(trans);
  const Eigen::Affine3d r(quat);
  
  Eigen::Matrix4d cam;
  cam << t.matrix() * r.matrix();

  pointsFromMsg(d_1 < d_2 ? cloud_msg_1 : cloud_msg_2, cam);

  // std::cout << "1: " << std::setprecision(17) << t << " vs " << cloud_msg->header.stamp.toSec() << std::endl;

}

void DataROS::depthCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  cache.add(cloud_msg);
  return;
}

void DataROS::pointsFromMsg(const sensor_msgs::PointCloud2ConstPtr& cloud_msg, Eigen::Matrix4d& cam)
{

  Eigen::MatrixXd points(cloud_msg->width*cloud_msg->height, 3);
  Eigen::MatrixXd colors(cloud_msg->width*cloud_msg->height, 3);

  // printf ("Cloud: width = %d, height = %d\n", cloud_msg->width, cloud_msg->height);
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  if (!operational) {
    Eigen::Affine3f r = findPlaneRotation(pcl_cloud);
    rOffset = r.matrix().cast<double>() * cam.inverse(); // inverse current camera
    operational = true;
  }

  cam = rOffset * cam;

  int rowcntr = 0;
#pragma omp parallel for
  for (uint y = 0; y < cloud_msg->height; ++y)
  {
    for (uint x = 0; x < cloud_msg->width; ++x)
    {
      const pcl::PointXYZRGB &pt = pcl_cloud->points[y * cloud_msg->width + x];
      // printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);

      if (x % 8 != 0 || y % 8 != 0)
      if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) continue;

      Eigen::Vector4d camera_point; camera_point << pt.x,pt.y,pt.z,1.0;
      camera_point = cam * camera_point;
      Eigen::RowVector4d world_point = camera_point.transpose();

#pragma omp critical
      {
        points.row(rowcntr) << world_point[0], -world_point[1], -world_point[2];
        colors.row(rowcntr) << (float)pt.r/255,(float)pt.g/255,(float)pt.b/255;
        ++rowcntr;
      }
    }
  }

  points.conservativeResize(rowcntr, Eigen::NoChange);
  colors.conservativeResize(rowcntr, Eigen::NoChange);

  if (subscriber != NULL && rowcntr > 0) {
    subscriber(points, colors);
  }
}

Eigen::Affine3f DataROS::findPlaneRotation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return Eigen::Affine3f::Identity();
  }

  // std::cout << "Model coefficients: " << coefficients->values[0] << " " 
  //                                     << coefficients->values[1] << " "
  //                                     << coefficients->values[2] << " " 
  //                                     << coefficients->values[3] << std::endl;

  // std::cout << inliers->indices.size() << std::endl;

  Eigen::Matrix<float, 1, 3> floor_plane_normal_vector, xy_plane_normal_vector;

  floor_plane_normal_vector[0] = coefficients->values[0];
  floor_plane_normal_vector[1] = coefficients->values[1];
  floor_plane_normal_vector[2] = coefficients->values[2];

  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 1.0;
  xy_plane_normal_vector[2] = 0.0;

  Eigen::Vector3f mu (0.0, 0.0, 0.0);

  for (uint i=0;i < inliers->indices.size();i++){

    // Get Point
    pcl::PointXYZRGB pt = cloud->points[inliers->indices[i]];

    mu[0] += pt.x;
    mu[1] += pt.y;
    mu[2] += pt.z;
  }

  mu /= (float)inliers->indices.size ();

  Eigen::Vector3f rotation_vector = xy_plane_normal_vector.cross(floor_plane_normal_vector);
  rotation_vector.normalize();
  float theta = -atan2f(rotation_vector.norm(), xy_plane_normal_vector.dot(floor_plane_normal_vector));

  Eigen::Affine3f transform_1(Eigen::Translation3f(-mu));

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  transform_2.rotate(Eigen::AngleAxisf (theta, rotation_vector));

  return transform_2 * transform_1;
}