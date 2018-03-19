#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>

#define MESH_RESOLUTION 100

class Mesh
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh();
  ~Mesh();
  
  void align_to_point_cloud(const Eigen::MatrixXd& P); // Basically resets the mesh
  void solve(const Eigen::MatrixXd& P); // Does nothing at this point
  
  const Eigen::MatrixXd& vertices();
  const Eigen::MatrixXi& faces();
  
private:
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::Vector3d mesh_normal;
};

#endif // MESH_HPP
