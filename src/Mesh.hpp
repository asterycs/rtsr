#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <ostream>

#include "EqHelpers.hpp"

#define MESH_RESOLUTION 5

class Mesh
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh();
  ~Mesh();
  
  void align_to_point_cloud(const Eigen::MatrixXd& P); // Basically resets the mesh
  void solve(const Eigen::MatrixXd& P);
  
  const Eigen::MatrixXd& vertices();
  const Eigen::MatrixXi& faces();
  
private:
  JtJMatrix<double> JtJ;
  JtzVector<double> Jtz;
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<double*> h;
  Eigen::Vector3d mesh_normal;
  Eigen::Matrix4d transform;
};

#endif // MESH_HPP
