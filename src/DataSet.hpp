#ifndef DATASET_HPP
#define DATASET_HPP

#include <fstream>
#include <experimental/filesystem>

#include "Eigen/Core"

class DataSet
{
public:
  DataSet(const std::string& folder);
  ~DataSet();
  
  bool get_next_point_cloud(Eigen::MatrixXd& P, Eigen::Matrix4d& world2camera);
private:
  bool getNextCamera(Eigen::Matrix4d& cam, const double timestamp);

  bool operational;
  std::vector<std::string> depth_files;
  int next_file_idx;
  std::string camera_ref_file_name;
};

#endif // DATASET_HPP
