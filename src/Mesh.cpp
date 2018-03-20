#include "Mesh.hpp"

#include "igl/embree/line_mesh_intersection.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"

#include <iostream>

template <typename Derived>
void remove_empty_rows(const Eigen::MatrixBase<Derived>& in, Eigen::MatrixBase<Derived> const& out_)
{
  Eigen::Matrix<bool, Eigen::Dynamic, 1> non_minus_one = (in.col(0) != -1);
  
  typename Eigen::MatrixBase<Derived>& out = const_cast<Eigen::MatrixBase<Derived>&>(out_);
  out.derived().resize(non_minus_one.count(), in.cols());

  typename Eigen::MatrixBase<Derived>::Index j = 0;
  for(typename Eigen::MatrixBase<Derived>::Index i = 0; i < in.rows(); ++i)
  {
    if (non_minus_one(i))
      out.row(j++) = in.row(i);
  }
}

Eigen::Matrix3d getBasis(const Eigen::Vector3d& n) {

  Eigen::Matrix3d R;

  Eigen::Vector3d Q = n;
  const Eigen::Vector3d absq = Q.cwiseAbs();

  Eigen::Matrix<int,1,1> min_idx;
  Eigen::Matrix<double,1,1> min_elem;
  
  igl::mat_min(absq, 1, min_elem, min_idx);
  
  Q(min_idx(0)) = 1;

  Eigen::Vector3d T = Q.cross(n).normalized();
  Eigen::Vector3d B = n.cross(T).normalized();

  R.col(0) = T;
  R.col(1) = B;
  R.col(2) = n;

  return R;
}


Mesh::Mesh()
{
  
}

Mesh::~Mesh()
{
  
}

void Mesh::align_to_point_cloud(const Eigen::MatrixXd& P)
{
  Eigen::RowVector3d plane_normal, plane_point;
  igl::fit_plane(P, plane_normal, plane_point);
  mesh_normal << plane_normal.transpose();
  
  const Eigen::RowVector3d bb_min = P.colwise().minCoeff();
  const Eigen::RowVector3d bb_max = P.colwise().maxCoeff();
  Eigen::RowVector3d bb_d = (bb_max - bb_min).cwiseAbs();
  
  const Eigen::Affine3d scaling(Eigen::Scaling(Eigen::Vector3d(bb_d(0)/MESH_RESOLUTION, 0.,bb_d(2)/MESH_RESOLUTION)));
  
  const Eigen::Vector3d P_centr = P.colwise().mean();
  const Eigen::Affine3d t(Eigen::Translation3d(P_centr - Eigen::Vector3d(0,0,0))); // Remove the zero vector if you dare ;)
  
  transform = t.matrix();
    
  V.resize(MESH_RESOLUTION*MESH_RESOLUTION, 3);
  F.resize((MESH_RESOLUTION-1)*(MESH_RESOLUTION-1)*2, 3);

#pragma omp parallel for
  for (int z_step = 0; z_step < MESH_RESOLUTION; ++z_step)
  {
    for (int x_step = 0; x_step < MESH_RESOLUTION; ++x_step)
    {
      Eigen::RowVector4d v; v << Eigen::RowVector3d(x_step-MESH_RESOLUTION/2,0.0,z_step-MESH_RESOLUTION/2),1.0;
      V.row(x_step + z_step*MESH_RESOLUTION) << (v * scaling.matrix().transpose() * transform.transpose()).head<3>();
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
}

  
const Eigen::MatrixXd& Mesh::vertices()
{
  return this->V;
}

const Eigen::MatrixXi& Mesh::faces()
{
  return this->F;
}

void Mesh::solve(const Eigen::MatrixXd& P)
{
  Eigen::MatrixXd points_to_interpolate = P;
  Eigen::MatrixXd normals = mesh_normal.transpose().replicate(P.rows(), 1);
  Eigen::MatrixXd bc = igl::embree::line_mesh_intersection(points_to_interpolate, normals, V, F);
  
  
}
