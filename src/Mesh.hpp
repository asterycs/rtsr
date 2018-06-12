#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/StdVector>
#include <Eigen/Geometry>

#include <cassert>
#include <ostream>

#include "EqHelpers.hpp"

// Number of vertices along one dimension in the base layer
#define MESH_RESOLUTION 5
// Scale factor of 1 makes the mesh the same size as the bounding box of the
// point cloud given to align_to_point_cloud
#define MESH_SCALING_FACTOR 1.55
// Multiresolution levels. 1 is a single resolution mesh
#define MESH_LEVELS 7
#define TEXTURE_RESOLUTION 300

struct Texture
{
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> red;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> green;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> blue;
};

struct ColorData
{
  Texture texture;
  Eigen::MatrixXd UV;
};

template <typename T>
class Mesh
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh();
  ~Mesh();

  void cleanup();
  
  template <typename Derived>
  void align_to_point_cloud(const Eigen::MatrixBase<Derived>& P);// Basically resets the mesh
  
  void set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P);
  void set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C);
  void solve(const int iterations);
  
  void get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out) const;
  void get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out, ColorData& colorData) const;
  
private:
  std::vector<JtJMatrixGrid<T>> JtJ;
  std::vector<JtzVector<T>> Jtz;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> current_target_point_cloud;
  Eigen::MatrixXi color_counter;
  Texture texture;
  
  std::vector<std::vector<std::vector<T>>> residuals; // For convergence analysis. [pc][level][iteration]
  
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>, Eigen::aligned_allocator<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> V; // Vertices
  std::vector<Eigen::MatrixXi, Eigen::aligned_allocator<Eigen::MatrixXi>> F; // Face vertex indices
  Eigen::Matrix<T, 4, 4> transform; // Mesh location and orientation

  void sor(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const;
  void sor_parallel(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const;
  
#ifdef ENABLE_CUDA
  void sor_gpu(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h);
#endif
  
  void project_points(const int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc) const;
  void update_JtJ(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc);
  void update_Jtz(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc, const Eigen::Matrix<T, Eigen::Dynamic, 1>& z);
};

#include "Mesh.tpp"

#endif // MESH_HPP
