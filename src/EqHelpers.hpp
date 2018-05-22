#ifndef EQHELPERS_HPP
#define EQHELPERS_HPP

#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <array>
#include "Util.hpp"

template <typename T>
class JtzVector
{  
public:
  CUDA_HOST JtzVector() = default;
  CUDA_HOST_DEVICE const T* get_vec() const
  {
    return this->vec;
  }
  
  void CUDA_HOST resize(const int mesh_width)
  {
	CUDA_CHECK(cudaFree(vec));
	CUDA_CHECK(cudaMallocManaged((void**)&vec, mesh_width*mesh_width));

	this->mesh_width = mesh_width;
  }
  
  void CUDA_HOST clear()
  {
    CUDA_CHECK(cudaFree(vec));
    this->mesh_width = 0;
  }

  void CUDA_HOST_DEVICE update_triangle(const int ti, const T a, const T b, const T val)
  {
    assert(ti < (mesh_width-1)*(mesh_width-1)*2);
    
    const int offset  = ti/2 % (mesh_width-1);
    const int multipl = ti/2 / (mesh_width-1);
    
    if (ti % 2 == 0)
    {
      vec[offset +     multipl * mesh_width]      += a * val;
      vec[offset + 1 + multipl * mesh_width]      += b * val;
      vec[offset +    (multipl+1) * mesh_width]   += (1 - a - b) * val;
    }else // ti % 2 == 1
    {
      vec[1 + offset +    (multipl+1) * mesh_width] += a * val;
      vec[1 + offset - 1 +(multipl+1) * mesh_width] += b * val;
      vec[1 + offset +   +(multipl)   * mesh_width] += (1 - a - b) * val;
    }
  }
  
private:
  int mesh_width;
  T* vec;
};

template <typename T>
class JtJMatrixGrid
{  
public:
  CUDA_HOST JtJMatrixGrid() = default;

  CUDA_HOST void resize(const int mesh_width)
  {     
	CUDA_CHECK(cudaFree(mat));
	CUDA_CHECK(cudaMallocManaged((void**)&mat, (2*mesh_width-1)*(2*mesh_width-1)));

    this->mesh_width = mesh_width;
    this->matrix_width = 2*mesh_width-1;
  }
  
  CUDA_HOST void clear()
  {
    CUDA_CHECK(cudaFree(mat));
    this->mesh_width = 0;
    this->matrix_width = 0;
  }

 CUDA_HOST_DEVICE unsigned int get_mesh_width() const
 {
   return this->mesh_width;
 }

 CUDA_HOST unsigned int get_matrix_width() const
 {
   return this->matrix_width;
 }
  
  CUDA_HOST const T* get_mat() const
  {
    return this->mat;
  }
  
  CUDA_HOST_DEVICE const T get_mat_value(const uint32_t r, const uint32_t c) const
  {
	  return mat[c + r*matrix_width];
  }

  CUDA_HOST_DEVICE void get_matrix_values_for_vertex(const int vi, T(&vals)[6], int(&ids)[6], T& a) const
  {
    const int vci = vi % mesh_width;
    const int vri = vi / mesh_width;
    
    const int c = vci * 2;
    const int r = vri * 2;
    
    // Clockwise ordering
        
    if (vci - 1 >= 0)
    {
      vals[0] = get_mat_value(r - 1, c);
      ids[0] = vi - 1;
    }else
    {
      vals[0] = 0.0;
      ids[0] = 0;
    }

    if (vci - 1 >= 0)
    {
      vals[1] = get_mat_value(r, c - 1);
      ids[1] = vi - mesh_width;
    }else
    {
      vals[1] = 0.0;
      ids[1] = 0;
    }

    if (vri + 1 < mesh_width && vci - 1 >= 0)
    {
      vals[2] = get_mat_value(r+1, c-1);
      ids[2] = vi - mesh_width + 1;
    }else
    {
      vals[2] = 0.0;
      ids[2] = 0;      
    }

    if (vri + 1 < mesh_width)
    {
      vals[3] = get_mat_value(r+1, c);
      ids[3] = vi + 1;
    }else
    {
      vals[3] = 0.0;
      ids[3] = 0;
    }
    
    if (vci + 1 < mesh_width)
    {
      vals[4] = get_mat_value(r, c+1);
      ids[4] = vi + mesh_width;
    }else
    {
      vals[4] = 0.0;
      ids[4] = 0;      
    }
    
    if (vri - 1 >= 0 && vci + 1 < mesh_width)
    {
      vals[5] = get_mat_value(r-1, c+1);
      ids[5] = vi + mesh_width - 1;     
    }else
    {
      vals[5] = 0.0;
      ids[5] = 0;         
    }

     a = get_mat_value(c, r);
  }
  
  /*
   void grow(const enum Direction d, steps) ...
    */
  
  // ti indexed as:
  //   __ __ __
  //  |0/|2/|4/|...
  //  |/1|/3|/5|
  //  |...
  CUDA_HOST_DEVICE void update_triangle(const int ti, const T a, const T b)
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
      
      update_if_present(vidx,vidy,                                               a*a);
      update_if_present(vidx+2,vidy,                                             b*b);
      update_if_present(vidx,vidy+2,                                 (1-a-b)*(1-a-b));
      update_if_present(vidx+1,vidy,                                             a*b);
      update_if_present(vidx+1,vidy+1,                                     (1-a-b)*b);
      update_if_present(vidx,vidy+1,                                       (1-a-b)*a);
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
      
      update_if_present(vidx,vidy,                                 a*a);
      update_if_present(vidx-2,vidy,                               b*b);
      update_if_present(vidx,vidy-2,                   (1-a-b)*(1-a-b));
      update_if_present(vidx-1,vidy,                               a*b);
      update_if_present(vidx,vidy-1,                         (1-a-b)*a);
      update_if_present(vidx-1,vidy-1,                       (1-a-b)*b);
    }
  }
  
private:
  CUDA_HOST_DEVICE void update_if_present(const int r, const int c, const T val)
  {
    if (r < matrix_width && c < matrix_width && c >= 0 && r >= 0)
      mat[r + c*matrix_width] += val;
  }
  
  int mesh_width;
  int matrix_width;
  T* mat;
};

#endif //EQHELPERS_HPP
