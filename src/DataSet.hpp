#ifndef DATASET_HPP
#define DATASET_HPP

#include <fstream>
#include <boost/filesystem.hpp>

#include "Eigen/Core"
#include <Eigen/Geometry> 


class DataSet
{
public:
  DataSet(const std::string& folder);
  ~DataSet();
  
  bool get_next_point_cloud(Eigen::MatrixXd& P, Eigen::MatrixXd &colors, Eigen::Matrix4d& t_camera);
  Eigen::Affine3f findPlaneRotation(Eigen::MatrixXd& points);
  void clip_point_clouds();
private:
  bool get_next_camera(Eigen::Matrix4d& cam, const double timestamp);
  std::string get_next_rgb(const double timestamp);

  bool operational;
  bool clip;
  std::vector<std::string> depth_files;
  int next_file_idx;
  std::string folder_path;
  std::string rgb_ref_file_name;
  std::fstream camera_ref_file;
  Eigen::Matrix4d rOffset;
};

#endif // DATASET_HPP
