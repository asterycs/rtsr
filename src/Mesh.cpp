#include "Mesh.hpp"

#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "igl/barycentric_to_global.h"

#include <Eigen/SparseQR>

#include <iostream>
#include <cstdlib>

template <typename T>
Mesh<T>::Mesh()
{
  
}

template <typename T>
Mesh<T>::~Mesh()
{
  
}

template <typename T>
template <int Rows, int Cols>
void Mesh<T>::align_to_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P)
{  
  using TvecR3 = Eigen::Matrix<T, 1, 3>;
  using TvecC3 = Eigen::Matrix<T, 3, 1>;
  
  const TvecR3 bb_min = P.colwise().minCoeff();
  const TvecR3 bb_max = P.colwise().maxCoeff();
  const TvecR3 bb_d = (bb_max - bb_min).cwiseAbs();
  
  // Scaling matrix
  const Eigen::Transform<T, 3, Eigen::Affine> scaling(Eigen::Scaling(TvecC3(MESH_SCALING_FACTOR*bb_d(0)/(MESH_RESOLUTION-1), 0.,MESH_SCALING_FACTOR*bb_d(2)/(MESH_RESOLUTION-1))));
  
  const TvecR3 pc_mean = P.colwise().mean();
  
  // P_centr: mean of the point cloud
  TvecR3 P_centr = bb_min + 0.5*(bb_max - bb_min);
  P_centr(1) = 0; // Move to mean height w/r to pc instead of bb.
  
  const Eigen::Transform<T, 3, Eigen::Affine> t(Eigen::Translation<T, 3>(P_centr.transpose()));
  
  transform = t.matrix();
    
  V.resize(MESH_RESOLUTION*MESH_RESOLUTION, 3);
  F.resize((MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2, 3);
  JtJ.resize(MESH_RESOLUTION);
  Jtz.resize(MESH_RESOLUTION);

#pragma omp parallel for
  for (int z_step = 0; z_step < MESH_RESOLUTION; ++z_step)
  {
    for (int x_step = 0; x_step < MESH_RESOLUTION; ++x_step)
    {
      Eigen::Matrix<T, 1, 4> v; v << TvecR3(x_step-T(MESH_RESOLUTION-1)/2.f,1.f, z_step-T(MESH_RESOLUTION-1)/2.f),1.f;
      V.row(x_step + z_step*MESH_RESOLUTION) << (v * scaling.matrix().transpose() * transform.transpose()).template head<3>();
    }
  }
  
#pragma omp parallel for
  for (int y_step = 0; y_step < MESH_RESOLUTION-1; ++y_step)
  {
    for (int x_step = 0; x_step < MESH_RESOLUTION-1; ++x_step)
    {
      F.row(x_step*2 + y_step*(MESH_RESOLUTION-1)*2)     << x_step+   y_step   *MESH_RESOLUTION,x_step+1+y_step*   MESH_RESOLUTION,x_step+(y_step+1)*MESH_RESOLUTION;
      F.row(x_step*2 + y_step*(MESH_RESOLUTION-1)*2 + 1) << x_step+1+(y_step+1)*MESH_RESOLUTION,x_step+ (y_step+1)*MESH_RESOLUTION,x_step+1+y_step*MESH_RESOLUTION;
    }
  }
  // Set constraints for each vertex:
  for (int i = 0; i < (MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2; ++i)
  {
    if (i % 2 == 0) {
      // Inside the grid only set 1 constraint per upper left triangle
      JtJ.update_triangle(i, 1.f, 0.f);
    } else if ((i + 1) % ((MESH_RESOLUTION-1)*2) == 0 || i >= (MESH_RESOLUTION-2)*(MESH_RESOLUTION-1)*2) {
      // last column or last row (set point for bottom right triangles too)
      JtJ.update_triangle(i, 1.f, 0.f);
    }
  }

  // two points on grid are not getting reached that way:
  JtJ.update_triangle((MESH_RESOLUTION-2)*2, 0.f, 1.f); // upper right
  JtJ.update_triangle((MESH_RESOLUTION-2)*(MESH_RESOLUTION-1)*2, 0.f, 0.f); // bottom left

}

template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& Mesh<T>::vertices()
{
  return this->V;
}

template <typename T>
const Eigen::MatrixXi& Mesh<T>::faces()
{
  return this->F;
}

template <typename T>
void barycentric(const Eigen::Matrix<T, 2, 1>& p, const Eigen::Matrix<T, 2, 1>& a, const Eigen::Matrix<T, 2, 1>& b, const Eigen::Matrix<T, 2, 1>& c, T &u, T &v, T &w)
{
    Eigen::Matrix<T, 2, 1> v0 = b - a,
                                           v1 = c - a,
                                           v2 = p - a;
    T a00 = v0.dot(v0);
    T a01 = v0.dot(v1);
    T a11 = v1.dot(v1);
    T a20 = v2.dot(v0);
    T a21 = v2.dot(v1);
    T den = 1 / (a00 * a11 - a01 * a01);
    v =  (a11 * a20 - a01 * a21) * den;
    w = (a00 * a21 - a01 * a20) * den;
    u = T(1.0) - v - w;
}

template <typename T, typename Derived>
void project_points(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V, Eigen::MatrixBase<Derived>& bc)
{
  bc.derived().resize(P.rows(), 3);
  
  // Upper left triangle. Used to compute the index of the triangle that is hit
  const Eigen::Matrix<T, 2, 1> V_00(V.row(0)(0), V.row(0)(2));
  const Eigen::Matrix<T, 2, 1> V_01(V.row(1)(0), V.row(1)(2));
  const Eigen::Matrix<T, 2, 1> V_10(V.row(MESH_RESOLUTION)(0), V.row(MESH_RESOLUTION)(2));
  
  const double dx = (V_01- V_00).norm();
  const double dy = (V_10 - V_00).norm();
  
#pragma omp parallel for
  for (int pi = 0; pi < P.rows(); ++pi)
  {
    const Eigen::Matrix<T, 2, 1> current_point(P.row(pi)(0), P.row(pi)(2));
    const Eigen::Matrix<T, 2, 1> offset( current_point - V_00 );
    
    const int c = static_cast<int>(offset(0) / dx);
    const int r = static_cast<int>(offset(1) / dy);
    
    if (c >= MESH_RESOLUTION-1 || r >= MESH_RESOLUTION-1)
    {
      bc.row(pi) << -1, T(0.0), T(0.0);
      continue;
    } 
    const  Eigen::Matrix<T, 3, 1> ul_3d = V.row(c + r*MESH_RESOLUTION);
    const  Eigen::Matrix<T, 3, 1> br_3d = V.row(c + 1 + (r+1)*MESH_RESOLUTION);
    
    const Eigen::Matrix<T, 2, 1> ul_reference(ul_3d(0), ul_3d(2));
    const Eigen::Matrix<T, 2, 1> br_reference(br_3d(0), br_3d(2));
    
    const double ul_squared_dist = (ul_reference - current_point).squaredNorm();
    const double br_squared_dist = (br_reference - current_point).squaredNorm();
    
    Eigen::Matrix<T, 2, 1> v_a;
    Eigen::Matrix<T, 2, 1> v_b;
    Eigen::Matrix<T, 2, 1> v_c;
    
    int f_idx = -1;
    
    if (ul_squared_dist <= br_squared_dist)
    {
      f_idx = 2 * c + r * 2 * (MESH_RESOLUTION-1);
      
      v_a << V.row(c + r*MESH_RESOLUTION)(0), V.row(c + r*MESH_RESOLUTION)(2);
      v_b << V.row(c+1 + r*MESH_RESOLUTION)(0),V.row(c+1 + r*MESH_RESOLUTION)(2);
      v_c << V.row(c + (r+1)*MESH_RESOLUTION)(0),V.row(c + (r+1)*MESH_RESOLUTION)(2);
    }else
    {
      f_idx = 2 * c + r * 2 * (MESH_RESOLUTION-1) + 1;
      
      v_a << V.row(c+1 + (r+1)*MESH_RESOLUTION)(0),V.row(c+1 + (r+1)*MESH_RESOLUTION)(2);
      v_b << V.row(c + (r+1)*MESH_RESOLUTION)(0),V.row(c + (r+1)*MESH_RESOLUTION)(2);
      v_c << V.row(c+1 + r*MESH_RESOLUTION)(0),V.row(c+1 + r*MESH_RESOLUTION)(2);
    }
    
    T u,v,w;
    barycentric(current_point, v_a, v_b, v_c, u, v, w);
    
    bc.row(pi) << f_idx, u, v;
  }
  
}

template <typename T>
template <int Rows, int Cols>
void Mesh<T>::set_target_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P)
{
  // Seems to be a bug in embree. tnear of a ray is not set corretly if vector is
  // along coordinate axis, needs slight offset.
  //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> normals = Eigen::Matrix<T, 1, 3>(0.0001, 1., 0.0).replicate(P.rows(), 1);
  //Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bc = igl::embree::line_mesh_intersection(P, normals, V, F);
  
  // Made our own projection function
  // bc stored as "face_index u v"
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bc;
  project_points<T>(P, V, bc);
  
  for (int i = 0; i < bc.rows(); ++i)
  {
    const Eigen::Matrix<T, 1, 3>& row = bc.row(i);

    if (static_cast<int>(row(0)) == -1)
      continue;
    
    JtJ.update_triangle(static_cast<int>(row(0)), row(1), row(2));
    Jtz.update_triangle(static_cast<int>(row(0)), row(1), row(2), P.row(i)(1));
  }
}

template <typename T>
void Mesh<T>::iterate()
{
  sor_parallel<1>(V.col(1));
}

// template <typename T>
// void Mesh<T>::qr_solve()
// {
//   Eigen::SparseMatrix<T>& JtJ_mat = JtJ.get_mat();
//   JtJ_mat.makeCompressed();

//   Eigen::SparseQR <Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int> > solver;

//   solver.compute(JtJ_mat);
//   if (solver.info() != Eigen::Success) {
//     std::cerr << "Decomposition failed" << std::endl;
//     return;
//   }

//   const Eigen::Matrix<T, Eigen::Dynamic, 1>& Jtz_vec = Jtz.get_vec();

//   Eigen::Matrix<T, Eigen::Dynamic, 1> h = solver.solve(Jtz_vec);
//   if (solver.info() != Eigen::Success) {
//     std::cerr << "Solving failed" << std::endl;
//     return;
//   }

//   V.col(1) << h;
// }

template <typename T>
template <int Iterations>
void Mesh<T>::sor(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz.get_vec();
    
  for(int it = 0; it < Iterations; it++)
  {
    for (int vi = 0; vi < h.rows(); vi++)
      sor_inner(vi, JtJ, Jtz_vec, h);
  }
}

template <typename T>
inline void sor_inner(const int vi, const JtJMatrixGrid<T>& JtJ, const Eigen::Matrix<T, Eigen::Dynamic, 1>& Jtz_vec, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h)
{
  T xn = Jtz_vec(vi);
  T acc = 0;

  std::array<double, 6> vals;
  std::array<int, 6> ids;

  double a;
  JtJ.get_matrix_values_for_vertex(vi, vals, ids, a);

  for (int j = 0; j < 6; ++j)
    acc += vals[j] * h(ids[j]);

  xn -= acc;

  const T w = 1.0;
  h(vi) = (1-w) * h(vi) + w*xn/a;
}

template <typename T>
template <int Iterations>
void Mesh<T>::sor_parallel(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz.get_vec();
    
  for(int it = 0; it < Iterations; it++)
  {
#pragma omp parallel for collapse(2)
    for (int x = 0; x < MESH_RESOLUTION; x+=2)
      for (int y = 0; y < MESH_RESOLUTION; y+=2)
      {
        const int vi = x + y * MESH_RESOLUTION;
        sor_inner(vi, JtJ, Jtz_vec, h);
      }

#pragma omp parallel for collapse(2)    
    for (int x = 1; x < MESH_RESOLUTION; x+=2)
      for (int y = 0; y < MESH_RESOLUTION; y+=2)
      {
        const int vi = x + y * MESH_RESOLUTION;
        sor_inner(vi, JtJ, Jtz_vec, h);
      }

#pragma omp parallel for collapse(2)
    for (int x = 0; x < MESH_RESOLUTION; x+=2)
      for (int y = 1; y < MESH_RESOLUTION; y+=2)
      {
        const int vi = x + y * MESH_RESOLUTION;
        sor_inner(vi, JtJ, Jtz_vec, h);
      }
      
#pragma omp parallel for collapse(2)
    for (int x = 1; x < MESH_RESOLUTION; x+=2)
      for (int y = 1; y < MESH_RESOLUTION; y+=2)
      {
        const int vi = x + y * MESH_RESOLUTION;
        sor_inner(vi, JtJ, Jtz_vec, h);
      }
  }
}


// Explicit instantiation
template class Mesh<double>;
template void Mesh<double>::align_to_point_cloud(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);
template void Mesh<double>::set_target_point_cloud(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);

//template class Mesh<float>;
//template void Mesh<float>::align_to_point_cloud(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&);
//template void Mesh<float>::solve(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&);
