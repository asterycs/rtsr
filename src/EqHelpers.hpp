#ifndef EQHELPERS_HPP
#define EQHELPERS_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <array>

template <typename T>
class JtzVector
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtzVector() = default;
  JtzVector(const int mesh_width) : mesh_width(mesh_width), vec(Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(mesh_width*mesh_width)) {};
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& get_vec() const
  {
    return this->vec;
  }
  
  void resize(const int mesh_width)
  {
    const typename Eigen::Matrix<T, Eigen::Dynamic, 1>::Index old_size = vec.rows();
    Eigen::Matrix<T, Eigen::Dynamic, 1> newv = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(mesh_width*mesh_width, 1);
    
    if (old_size > 0)
      newv.head(old_size) = vec; 
      
    vec = newv;
    this->mesh_width = mesh_width;
  }
  
  void update_triangle(const int ti, const T a, const T b, const T val)
  {
    assert(ti < (mesh_width-1)*(mesh_width-1)*2);
    
    const int offset  = ti/2 % (mesh_width-1);
    const int multipl = ti/2 / (mesh_width-1);
    
    if (ti % 2 == 0)
    {
      vec(offset +     multipl * mesh_width)      += a * val;
      vec(offset + 1 + multipl * mesh_width)      += b * val;
      vec(offset +    (multipl+1) * mesh_width)   += (1 - a - b) * val;
    }else // ti % 2 == 1
    {
      vec(1 + offset +    (multipl+1) * mesh_width) += a * val;
      vec(1 + offset - 1 +(multipl+1) * mesh_width) += b * val;
      vec(1 + offset +   +(multipl)   * mesh_width) += (1 - a - b) * val; 
    }
  }
  
private:
  int mesh_width;
  Eigen::Matrix<T, Eigen::Dynamic, 1> vec;
};

template <typename T>
class [[deprecated]] JtJMatrix // Could be made into a Eigen::SparseMatrix later
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtJMatrix() = default;
  JtJMatrix(const int mesh_width) : mesh_width(mesh_width), mat(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(mesh_width*mesh_width, mesh_width*mesh_width)) {};
  void resize(const int mesh_width)
  {
    mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(mesh_width*mesh_width, mesh_width*mesh_width);
    this->mesh_width = mesh_width;
  }
  
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& get_mat() const
  {
    return this->mat;
  }

  // ti indexed as:
  //   __ __ __
  //  |0/|2/|4/|...
  //  |/1|/3|/5|
  //  |...
  void update_triangle(const int ti, const T a, const T b)
  {
    assert(ti < (mesh_width-1)*(mesh_width-1)*2);
        
    // Triangle is here (*):
    //  __ __ 
    // |*/| /..
    // |‾‾|‾‾
    // ...
    if (ti % 2 == 0)
    {     
      const int vxidx = (ti % (2*(mesh_width - 1))) / 2;
      const int vyidx =  ti / (2*(mesh_width - 1));

      const int vidx = vxidx + vyidx * mesh_width;
      
      update(vidx,vidx,                                       a*a);
      update(vidx+1,vidx+1,                                   b*b);
      update(vidx+mesh_width,vidx+mesh_width,     (1-a-b)*(1-a-b));

      assert(vxidx < mesh_width-1);
        update(vidx,vidx+1,                                   a*b);
        
      assert(vxidx < mesh_width-1 && vyidx < mesh_width-1);
        update(vidx+1,vidx+mesh_width,                       (1-a-b)*b);

      assert(vyidx < mesh_width-1);
        update(vidx,vidx+mesh_width,                         (1-a-b)*a);
    }else
    // ti % 2 == 1
    // Triangle is here (*):
    //  __ __ 
    // |/*| /..
    // |‾‾|‾‾
    // ...
    {
      const int vxidx = (ti % (2*(mesh_width - 1))) / 2 + 1;
      const int vyidx =  ti / (2*(mesh_width - 1)) + 1;

      const int vidx = vxidx + vyidx * mesh_width;
      
      update(vidx,vidx,                                   a*a);
      update(vidx-1,vidx-1,                               b*b);
      update(vidx-mesh_width,vidx-mesh_width, (1-a-b)*(1-a-b));

      assert(vxidx > 0);
        update(vidx,vidx-1,                        a*b);
        
      assert(vxidx > 0 && vyidx > 0);
        update(vidx-mesh_width,vidx,       (1-a-b)*a);

      assert(vyidx > 0);
        update(vidx-mesh_width,vidx-1,       (1-a-b)*b); 
    }
  }
  
private:
  void update(const int r, const int c, const T val)
  {
    if (r < mat.rows() && c < mat.cols())
      mat(r,c) += val;
      
    if (r != c)
      if (c < mat.rows() && r < mat.cols())
        mat(c,r) += val;
  }
  
  int mesh_width;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
};

template <typename T>
class JtJMatrixGrid
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtJMatrixGrid() = default;
  JtJMatrixGrid(const int mesh_width) : mesh_width(mesh_width), mat(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero((2*mesh_width-1)*(2*mesh_width-1), (2*mesh_width-1)*(2*mesh_width-1))) {};
  void resize(const int mesh_width)
  {     
    mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero((2*mesh_width-1), (2*mesh_width-1));
    this->mesh_width = mesh_width;
  }
  
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& get_mat() const
  {
    return this->mat;
  }
  
  void get_matrix_values_for_vertex(const int vi, std::array<T, 6>& vals, std::array<int, 6>& ids, T& a) const
  {
    const int vxi = vi % mesh_width;
    const int vyi = vi / mesh_width;
    
    const int x = vxi * 2;
    const int y = vyi * 2;
    
    // Clockwise ordering
        
    if (vxi - 1 >= 0)
    {
      vals[0] = mat(x - 1, y);
      ids[0] = vi - 1;
    }else
    {
      vals[0] = 0.0;
      ids[0] = 0;
    }

    if (vyi - 1 >= 0)
    {
      vals[1] = mat(x, y - 1);
      ids[1] = vi - mesh_width;
    }else
    {
      vals[1] = 0.0;
      ids[1] = 0;
    }

    if (vxi + 1 < mesh_width && vyi - 1 >= 0)
    {
      vals[2] = mat(x+1, y-1);
      ids[2] = vi - mesh_width + 1;
    }else
    {
      vals[2] = 0.0;
      ids[2] = 0;      
    }

    if (vxi + 1 < mesh_width)
    {
      vals[3] = mat(x+1, y);
      ids[3] = vi + 1;
    }else
    {
      vals[3] = 0.0;
      ids[3] = 0;
    }
    
    if (vyi +1 < mesh_width)
    {
      vals[4] = mat(x, y+1);
      ids[4] = vi + mesh_width;
    }else
    {
      vals[4] = 0.0;
      ids[4] = 0;      
    }
    
    if (vxi - 1 >= 0 && vyi + 1 < mesh_width)
    {
      vals[5] = mat(x-1, y+1);
      ids[5] = vi + mesh_width - 1;     
    }else
    {
      vals[5] = 0.0;
      ids[5] = 0;         
    }

     a = mat(x, y);
  }
  
  /*
   void grow(const enum Direction d, steps) ...
    */
  
  // ti indexed as:
  //   __ __ __
  //  |0/|2/|4/|...
  //  |/1|/3|/5|
  //  |...
  void update_triangle(const int ti, const T a, const T b)
  {
    assert(ti < (mesh_width-1)*(mesh_width-1)*2);
        
    // Triangle is here (*):
    //  __ __ 
    // |*/| /..
    // |‾‾|‾‾
    // ...
    if (ti % 2 == 0)
    {     
      const int vxidx = (ti % (2*(mesh_width - 1))) / 2; // Vertex index on the mesh
      const int vyidx =  ti / (2*(mesh_width - 1));

      const int vidx = vxidx*2; // Index in the grid matrix, Figure 5(c) in the paper
      const int vidy = vyidx*2;
      
      update_if_present(vidx,vidy,                                                                     a*a);
      update_if_present(vidx+2,vidy,                                                                b*b);
      update_if_present(vidx,vidy+2,                                               (1-a-b)*(1-a-b));
      update_if_present(vidx+1,vidy,                                                                 a*b);
      update_if_present(vidx+1,vidy+1,                                                   (1-a-b)*b);
      update_if_present(vidx,vidy+1,                                                        (1-a-b)*a);
    }else
    // ti % 2 == 1
    // Triangle is here (*):
    //  __ __ 
    // |/*| /..
    // |‾‾|‾‾
    // ...
    {
      const int vxidx = (ti % (2*(mesh_width - 1))) / 2 + 1;
      const int vyidx =  ti / (2*(mesh_width - 1)) + 1;

      const int vidx = vxidx*2;
      const int vidy = vyidx*2;
      
      update_if_present(vidx,vidy,                                   a*a);
      update_if_present(vidx-2,vidy,                               b*b);
      update_if_present(vidx,vidy-2,              (1-a-b)*(1-a-b));
      update_if_present(vidx-1,vidy,                                a*b);
      update_if_present(vidx,vidy-1,                       (1-a-b)*a);
      update_if_present(vidx-1,vidy-1,                   (1-a-b)*b); 
    }
  }
  
private:
  void update_if_present(const int r, const int c, const T val)
  {
    if (r < mat.rows() && c < mat.cols())
      mat(r,c) += val;
  }
  
  int mesh_width;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
};

#endif //EQHELPERS_HPP