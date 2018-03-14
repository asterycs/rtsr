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
  bool getNextCamera(Eigen::Matrix4d& cam);

  bool operational;
  std::experimental::filesystem::directory_iterator depth_it;
  std::ifstream camera_ref_file;
};

#endif // DATASET_HPP
