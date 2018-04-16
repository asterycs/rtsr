#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <ostream>

#include "EqHelpers.hpp"

// Number of vertices along one dimension
#define MESH_RESOLUTION 50
// Scale factor. 1 makes the mesh the same size as the bb of the
// pc given to align_to_point_cloud
#define MESH_SCALING_FACTOR 1.2

template <typename T>
class Mesh
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh();
  ~Mesh();
  
  template <int Rows, int Cols>
  void align_to_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P);// Basically resets the mesh
  
  template <int Rows, int Cols>
  void set_target_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P);
  void iterate();
  
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& vertices();
  const Eigen::MatrixXi& faces();
  
private:
  JtJMatrixGrid<T> JtJ;
  JtzVector<T> Jtz;
  
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V; // Vertices
  Eigen::MatrixXi F; // Face vertex indices
  Eigen::Matrix<T, 4, 4> transform; // Mesh location and orientation

  void gauss_seidel(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h, int iterations) const;
};

#endif // MESH_HPP
