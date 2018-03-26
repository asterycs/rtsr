#ifndef EQHELPERS_HPP
#define EQHELPERS_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <iostream>

template <typename T>
class JtzVector
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtzVector() = default;
  JtzVector(const int mesh_width) : mesh_width(mesh_width), vec(Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(mesh_width*mesh_width)) {};
  const Eigen::Matrix<T, Eigen::Dynamic, 1>& get_vec()
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
    
    updates.resize(mesh_width*mesh_width);
  }
  
  void update_triangle(const int ti, const T a, const T b, const T val)
  {
    assert(ti < (mesh_width-1)*(mesh_width-1)*2);
    
    ++updates(ti);
    const T old_factor = (updates(ti)-1.f)/updates(ti);
    
    const int offset  = ti/2 % (mesh_width-1);
    const int multipl = ti/2 / (mesh_width-1);
    
    if (ti % 2 == 0)
    {
      vec(offset +     multipl * mesh_width)      = old_factor * vec(offset + multipl * mesh_width) + (1-old_factor) * a * val;
      vec(offset + 1 + multipl * mesh_width)      = old_factor * vec(offset + 1 + multipl * mesh_width) + (1-old_factor) * b * val;
      vec(offset +    (multipl+1) * mesh_width)   = old_factor * vec(offset +    (multipl+1) * mesh_width) + (1-old_factor) * (1 - a - b) * val;
    }else // ti % 2 == 1
    {
      vec(1 + offset +    (multipl+1) * mesh_width) = old_factor * vec(1 + offset +    (multipl+1) * mesh_width) + (1-old_factor) * a * val;
      vec(1 + offset - 1 +(multipl+1) * mesh_width) = old_factor * vec(1 + offset - 1 +(multipl+1) * mesh_width) + (1-old_factor) * b * val;
      vec(1 + offset +   +(multipl)   * mesh_width) = old_factor * vec(1 + offset +   +(multipl)   * mesh_width) + (1-old_factor) * (1 - a - b) * val; 
    }
  }
  
private:
  int mesh_width;
  Eigen::Matrix<T, Eigen::Dynamic, 1> vec;
  Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> updates;
};

template <typename T>
class JtJMatrix // Could be made into a Eigen::SparseMatrix later
{  
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  JtJMatrix() = default;
  JtJMatrix(const int mesh_width) : mesh_width(mesh_width), mat(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(mesh_width*mesh_width, mesh_width*mesh_width)) {};
  void resize(const int mesh_width)
  {
    const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index old_r = mat.rows();
    const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index old_c = mat.cols();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> newm = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(mesh_width*mesh_width, mesh_width*mesh_width);
    
    if (old_r > 0 && old_c > 0)
      newm.block(0,0,old_r,old_c) = mat; 
      
    mat = newm;
    this->mesh_width = mesh_width;
    
    updates.resize(mesh_width*mesh_width,mesh_width*mesh_width);
  }
  
  const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& get_mat()
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

      if (vxidx < mesh_width-1)
        update(vidx,vidx+1,                                   a*b);
        
      if (vxidx < mesh_width-1 && vyidx < mesh_width-1)
        update(vidx+1,vidx+mesh_width,                       (1-a-b)*b);

      if (vyidx < mesh_width-1)
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

      if (vxidx > 0)
        update(vidx,vidx-1,                        a*b);
        
      if (vxidx > 0 && vyidx > 0)
        update(vidx-mesh_width,vidx,       (1-a-b)*a);

      if (vyidx > 0)
        update(vidx-mesh_width,vidx-1,       (1-a-b)*b); 
    }
  }
  
private:
  void update(const int r, const int c, const T val)
  {
    ++updates(r,c);
    const T old_factor = (updates(r,c)-1.f)/updates(r,c);
    if (r < mat.rows() && c < mat.cols())
      mat(r,c) = old_factor * mat(r,c) + (1-old_factor) * val;
      
    if (r != c)
      if (c < mat.rows() && r < mat.cols())
        mat(c,r) = old_factor * mat(c,r) + (1-old_factor) * val;
  }
  
  int mesh_width;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
  Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> updates;
};

#endif //EQHELPERS_HPP