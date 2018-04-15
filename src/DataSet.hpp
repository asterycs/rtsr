#ifndef DATASET_HPP
#define DATASET_HPP

#include <fstream>
#include <boost/filesystem.hpp>

#include "Eigen/Core"

class DataSet
{
public:
  DataSet(const std::string& folder);
  ~DataSet();
  
  bool get_next_point_cloud(Eigen::MatrixXd& P, Eigen::MatrixXd &colors, Eigen::Matrix4d& t_camera);
private:
  bool get_next_camera(Eigen::Matrix4d& cam, const double timestamp);
  const char* get_next_rgb(const double timestamp);

  bool operational;
  std::vector<std::string> depth_files;
  int next_file_idx;
  std::string folder_path;
  std::string camera_ref_file_name;
  std::string rgb_ref_file_name;
};

#endif // DATASET_HPP
