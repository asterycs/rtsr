#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>

#define MESH_RESOLUTION 100

struct SymmetricMatrix // Could be made into a Eigen::SparseMatrix later
{
  Eigen::MatrixXd mat;
  
  SymmetricMatrix(const int rows, const int cols) : mat(Eigen::MatrixXd::Zero(rows, cols)) {};
  
  void update(const int r, const int c, const double val)
  {
    if (r < mat.rows() && c < mat.cols())
      mat(r,c) = val;
    if (c < mat.rows() && r < mat.cols())
      mat(c,r) = val;
  }

  void update_vertex(const int vi, const double a, const int b)
  {
    update(vi,vi  ,               a*a);
    update(vi,vi+1,               a*b);
    update(vi,vi+2,         (1-a-b)*a);

    update(vi+1,vi+1,             b*b);
    update(vi+1,vi+2,       (1-a-b)*b);
    
    update(vi+2,vi+2, (1-a-b)*(1-a-b));
  }
};

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
  Eigen::Matrix4d transform;
};

#endif // MESH_HPP
