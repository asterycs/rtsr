#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <cassert>
#include <ostream>

#include "EqHelpers.hpp"

// Number of vertices along one dimension
#define MESH_RESOLUTION 2
// Scale factor. 1 makes the mesh the same size as the bounding box of the
// point cloud given to align_to_point_cloud
#define MESH_SCALING_FACTOR 1.2
#define MESH_LEVELS 5

template <typename T>
class Mesh
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh();
  ~Mesh();
  
  template <int Rows, int Cols>
  void align_to_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P);// Basically resets the mesh
  
  void set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P);
  void solve(const int iterations);
  
  void get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out) const;
  
private:
  std::vector<JtJMatrixGrid<T>, Eigen::aligned_allocator<JtJMatrixGrid<T>>> JtJ;
  std::vector<JtzVector<T>, Eigen::aligned_allocator<JtzVector<T>>> Jtz;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> current_target_point_cloud;
  
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> V; // Vertices
  std::vector<Eigen::MatrixXi, Eigen::aligned_allocator<Eigen::MatrixXi>> F; // Face vertex indices
  Eigen::Matrix<T, 4, 4> transform; // Mesh location and orientation

  void sor(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const;
  void sor_parallel(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const;
 void parallel_gpu_solve(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h);
  
  void project_points(const int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc) const;
  void update_weights(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc, const Eigen::Matrix<T, Eigen::Dynamic, 1>& z);
};

#include "Mesh.tpp"

#endif // MESH_HPP
