#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <ostream>

#include "EqHelpers.hpp"

#define MESH_RESOLUTION 2

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
  void solve(const Eigen::Matrix<T, Rows, Cols>& P);
  
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& vertices();
  const Eigen::MatrixXi& faces();
  
private:
  JtJMatrix<T> JtJ;
  JtzVector<T> Jtz;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V; // Vertices
  Eigen::MatrixXi F; // Face vertex indices
  std::vector<T*> h; // Pointers to the y coord in vertices that are solved in the linear system
  Eigen::Matrix<T, 4, 4> transform; // Mesh location and orientation
};

#endif // MESH_HPP
