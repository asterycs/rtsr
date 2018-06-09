#ifndef EQHELPERS_HPP
#define EQHELPERS_HPP

#include <cassert>
#include <iostream>
#include <array>

#include "Util.hpp"


template <typename T>
class JtzVector
{  
public:
  JtzVector() = default;
  
  CUDA_HOST void resize(const int mesh_width);
  CUDA_HOST void clear();
  CUDA_HOST_DEVICE T get(const int idx) const;
  CUDA_HOST void update_triangle(const int ti, const T a, const T b, const T val);
  
private:
  int mesh_width;
  T* vec;
};

template <typename T>
class JtJMatrixGrid
{  
private:
    CUDA_HOST_DEVICE T& get(const int r, const int c) const;

public:
    JtJMatrixGrid() = default;

    CUDA_HOST void resize(const int mesh_width);
    CUDA_HOST void clear();
    CUDA_HOST_DEVICE const T* get_matrix() const;
    CUDA_HOST_DEVICE unsigned int get_mesh_width() const;
    CUDA_HOST_DEVICE unsigned int get_matrix_width() const;
    CUDA_HOST_DEVICE void get_matrix_values_for_vertex(const int vi, T(&vals)[6], int(&ids)[6], T& a) const;
  
    // ti indexed as:
    //   __ __ __
    //  |0/|2/|4/|...
    //  |/1|/3|/5|
    //  |...
    CUDA_HOST void update_triangle(const int ti, const T a, const T b);
  
private:
  CUDA_HOST void update_if_present(const int r, const int c, const T val);
  
  int matrix_width;
  int mesh_width;
  T* mat;
};

#include "EqHelpers.tpp"

#endif //EQHELPERS_HPP
