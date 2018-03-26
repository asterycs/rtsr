#include "Mesh.hpp"

#include "igl/embree/line_mesh_intersection.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "igl/barycentric_to_global.h"

#include <Eigen/QR>

#include <iostream>
#include <cstdlib>

template <typename A, typename B>
Eigen::Matrix<typename B::Scalar, A::RowsAtCompileTime, A::ColsAtCompileTime> 
  extract(const Eigen::DenseBase<B>& full, const Eigen::DenseBase<A>& ind)
{
  using target_t = Eigen::Matrix <typename B::Scalar, A::RowsAtCompileTime, A::ColsAtCompileTime>;
  int num_indices = ind.innerSize();
  target_t target(num_indices);
  
#pragma omp parallel for
  for (int i = 0; i < num_indices; ++i)
    target[i] = full[ind[i]];
  
  return target;
} 

template <typename Derived>
void remove_empty_rows(const Eigen::MatrixBase<Derived>& in, Eigen::MatrixBase<Derived> const& out_)
{
  Eigen::Matrix<bool, Eigen::Dynamic, 1> non_minus_one(in.rows(), 1);
  
#pragma omp parallel for
  for (int i = 0; i < in.rows(); ++i)
  {
    if (in.row(i).isApprox(Eigen::Matrix<typename Derived::RealScalar, 1, 3>(-1., 0., 0.)))
      non_minus_one(i) = false;
    else
      non_minus_one(i) = true;
  }
  
  typename Eigen::MatrixBase<Derived>& out = const_cast<Eigen::MatrixBase<Derived>&>(out_);
  out.derived().resize(non_minus_one.count(), in.cols());

  typename Eigen::MatrixBase<Derived>::Index j = 0;
  for(typename Eigen::MatrixBase<Derived>::Index i = 0; i < in.rows(); ++i)
  {
    if (non_minus_one(i))
      out.row(j++) = in.row(i);
  }
}

template <typename InType>
Eigen::Matrix<InType, 3, 1> get_basis(const Eigen::Matrix<InType, 3, 1>& n) {

  Eigen::Matrix<InType, 3, 3> R;

  Eigen::Matrix<InType, 3, 1> Q = n;
  const Eigen::Matrix<InType, 3, 1> absq = Q.cwiseAbs();

  Eigen::Matrix<int,1,1> min_idx;
  Eigen::Matrix<InType,1,1> min_elem;
  
  igl::mat_min(absq, 1, min_elem, min_idx);
  
  Q(min_idx(0)) = 1;

  Eigen::Matrix<InType, 3, 1> T = Q.cross(n).normalized();
  Eigen::Matrix<InType, 3, 1> B = n.cross(T).normalized();

  R.col(0) = T;
  R.col(1) = B;
  R.col(2) = n;

  return R;
}

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
  const Eigen::Matrix<T, 1, 3> bb_min = P.colwise().minCoeff();
  const Eigen::Matrix<T, 1, 3> bb_max = P.colwise().maxCoeff();
  Eigen::Matrix<T, 1, 3> bb_d = (bb_max - bb_min).cwiseAbs();
  
  const Eigen::Transform<T, 3, Eigen::Affine> scaling(Eigen::Scaling(Eigen::Matrix<T, 3, 1>(bb_d(0)/(MESH_RESOLUTION-1), 0.,bb_d(2)/(MESH_RESOLUTION-1))));
  
  const Eigen::Matrix<T, 1, 3> P_centr = bb_min + 0.5*(bb_max - bb_min);
  const Eigen::Transform<T, 3, Eigen::Affine> t(Eigen::Translation<T, 3>(P_centr - Eigen::Matrix<T, 1, 3>(0,0,0))); // Remove the zero vector if you dare ;)
  
  transform = t.matrix();
    
  h.resize(MESH_RESOLUTION*MESH_RESOLUTION);
  V.resize(MESH_RESOLUTION*MESH_RESOLUTION, 3);
  F.resize((MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2, 3);
  JtJ.resize(MESH_RESOLUTION);
  Jtz.resize(MESH_RESOLUTION);

#pragma omp parallel for
  for (int z_step = 0; z_step < MESH_RESOLUTION; ++z_step)
  {
    for (int x_step = 0; x_step < MESH_RESOLUTION; ++x_step)
    {
      Eigen::Matrix<T, 1, 4> v; v << Eigen::Matrix<T, 1, 3>(x_step-(MESH_RESOLUTION-1)/2.f,1.f,z_step-(MESH_RESOLUTION-1)/2.f),1.f;
      V.row(x_step + z_step*MESH_RESOLUTION) << (v * scaling.matrix().transpose() * transform.transpose()).template head<3>();
      
      h[x_step + z_step*MESH_RESOLUTION] = &V.row(x_step + z_step*MESH_RESOLUTION)(1);
    }
  }
  
#pragma omp parallel for
  for (int y_step = 0; y_step < MESH_RESOLUTION-1; ++y_step)
  {
    for (int x_step = 0; x_step < MESH_RESOLUTION-1; ++x_step)
    {
      // JtJ matrix implementation depends on this indexing, if you hange this you need to change the JtJ class.
      F.row(x_step*2 + y_step*(MESH_RESOLUTION-1)*2)     << x_step+   y_step   *MESH_RESOLUTION,x_step+1+y_step*   MESH_RESOLUTION,x_step+(y_step+1)*MESH_RESOLUTION;
      F.row(x_step*2 + y_step*(MESH_RESOLUTION-1)*2 + 1) << x_step+1+(y_step+1)*MESH_RESOLUTION,x_step+ (y_step+1)*MESH_RESOLUTION,x_step+1+y_step*MESH_RESOLUTION;
    }
  }
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
template <int Rows, int Cols>
void Mesh<T>::solve(const Eigen::Matrix<T, Rows, Cols>& P)
{
  //using ScalType = typename Derived::RealScalar;
  // Seems to be a bug in embree. tnear of a ray is not set corretly if vector is
  // along coordinate axis, needs slight offset.
  //Eigen::Matrix<ScalType, Eigen::Dynamic, Eigen::Dynamic> normals = Eigen::Matrix<ScalType, 1, 3>(0.0001, 1., 0.0).replicate(P.rows(), 1);
  
  //Eigen::Matrix<ScalType, Eigen::Dynamic, Eigen::Dynamic> bc = igl::embree::line_mesh_intersection(P.template cast<ScalType>(), normals, V, F);
  
  //Eigen::Matrix<ScalType, Eigen::Dynamic, Eigen::Dynamic> filtered_bc;
  //remove_empty_rows(bc, filtered_bc);
  
  for (int i = 0; i < (MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2; ++i)
  {
    //const Eigen::Matrix<ScalType, 1, 3>& row = filtered_bc.row(i);
    JtJ.update_triangle(i, 0.33f, 0.33f);
  }
  
  //JtJ.update_triangle(1, 0.33, 0.33); 
  //JtJ.update_triangle(1, 0.33, 0.33); 

  //Jtz.update_triangle(0, 0.33, 0.33, 1);
  //Jtz.update_triangle(1, 0.33, 0.33, 1);
  
  for (int i = 0; i < (MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2; ++i)
  {
    Jtz.update_vertex(i, 0.33f, 0.33f, 1.f);
  }
    
  std::cout << JtJ.get_mat() << std::endl << std::endl;
  std::cout << Jtz.get_vec() << std::endl;
  
  //Eigen::VectorXd res = JtJ.get_mat().colPivHouseholderQr().solve(Jtz.get_vec());
  //std::cout << res << std::endl;
  
  //int cntr = 0;
  //for (T* hp : h)
  //  *hp = res(cntr++);
}

template <typename T>
template <int JtJRows, int JtJCols, int JtzRows>
void Mesh<T>::gauss_seidel(const Eigen::Matrix<T, JtJRows, JtJCols>& JtJ, Eigen::Matrix<T, JtzRows, 1>& h, Eigen::Matrix<T, JtzRows, 1>& Jtz, int iterations)
{    
    for(int i = 0; i < iterations; i++)
    {
        for (int v = 0; v < h.rows(); v++)
        {
            double xn = Jtz(v);
            for (int x = 0; x < JtJ.rows(); x++)
            {
                for (int y = 0; y < JtJ.rows(); y++)
                {
                    xn -= h(x)*JtJ(x, y);
                }
            }
            h(v) = xn / JtJ(v, v);
        }
    }
}
// Explicit instantiation
template class Mesh<double>;
template void Mesh<double>::align_to_point_cloud(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);
template void Mesh<double>::solve(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);
template void Mesh<double>::gauss_seidel(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& JtJ, Eigen::Matrix<double, Eigen::Dynamic, 1>& h, Eigen::Matrix<double, Eigen::Dynamic, 1>& Jtz, int iterations);

//template class Mesh<float>;
//template void Mesh<float>::align_to_point_cloud(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&);
//template void Mesh<float>::solve(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>&);