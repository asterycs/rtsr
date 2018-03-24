#ifndef MESH_HPP
#define MESH_HPP

#include <Eigen/Geometry>
#include <cassert>

#include <ostream>

#define MESH_RESOLUTION 5

class JtzVector
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtzVector(const int vertices) : vec(Eigen::VectorXd::Zero(vertices)) {};
  
  void update_triangle(const int ti, const double a, const int b)
  {
    // TODO
  }
  
private:
  Eigen::VectorXd vec;
};

class JtJMatrix // Could be made into a Eigen::SparseMatrix later
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtJMatrix(const int rows, const int cols) : mat(Eigen::MatrixXd::Zero(rows, cols)) {};
  const Eigen::MatrixXd& get_mat()
  {
    return this->mat;
  }

  // Discards old value. Should compute mean i believe...
  // ti indexed as:
  //   __ __ __
  //  |0/|2/|4/|...
  //  |/1|/3|/5|
  //  |...
  void update_triangle(const int ti, const double a, const int b)
  {
    assert(ti < (MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2);
        
    // Triangle is here (*):
    //  __ __ 
    // |*/| /..
    // |‾‾|‾‾
    // ...
    if (ti % 2 == 0)
    {     
      const int vxidx = (ti % (2*(MESH_RESOLUTION - 1))) / 2;
      const int vyidx =  ti / (2*(MESH_RESOLUTION - 1));

      const int vidx = vxidx + vyidx * MESH_RESOLUTION;
      
      update(vidx,vidx,                                                 a*a);

      if (vxidx < MESH_RESOLUTION-1)
        update(vidx,vidx+1,                                             a*b);
        
      if (vxidx < MESH_RESOLUTION-1 && vyidx < MESH_RESOLUTION-1)
        update(vidx+1,vidx+MESH_RESOLUTION,                       (1-a-b)*b);

      if (vyidx < MESH_RESOLUTION-1)
        update(vidx,vidx+MESH_RESOLUTION,                         (1-a-b)*a);
 
    }else
    // ti % 2 == 1
    // Triangle is here (*):
    //  __ __ 
    // |/*| /..
    // |‾‾|‾‾
    // ...
    {
      const int vxidx = (ti % (2*(MESH_RESOLUTION - 1))) / 2 + 1;
      const int vyidx =  ti / (2*(MESH_RESOLUTION - 1)) + 1;

      const int vidx = vxidx + vyidx * MESH_RESOLUTION;
      
      update(vidx,vidx,                                 a*a);

      if (vxidx > 0)
        update(vidx,vidx-1,                             a*b);
        
      if (vxidx > 0 && vyidx > 0)
        update(vidx-1,vidx+MESH_RESOLUTION,       (1-a-b)*b);

      if (vyidx > 0)
        update(vidx,vidx-MESH_RESOLUTION,         (1-a-b)*a); 
    }
  }
  
private:
  void update(const int r, const int c, const double val)
  {
    if (r < mat.rows() && c < mat.cols())
      mat(r,c) = val;
    if (c < mat.rows() && r < mat.cols())
      mat(c,r) = val;
  }
  
  Eigen::MatrixXd mat;
};

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
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::vector<double*> h;
  Eigen::Vector3d mesh_normal;
  Eigen::Matrix4d transform;
};

#endif // MESH_HPP
