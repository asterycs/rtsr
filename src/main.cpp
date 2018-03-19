#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "igl/embree/line_mesh_intersection.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "DataSet.hpp"

#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>

DataSet* global_ds_ptr; // Dirty dirty...

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

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) {
    
  Eigen::MatrixXd points;
  Eigen::Matrix4d world2cam;
  
  if (!global_ds_ptr->get_next_point_cloud(points, world2cam))
  {
    std::cerr << "Couldn't read data" << std::endl;
    return EXIT_FAILURE;
  }

  viewer.data().add_points(points, Eigen::RowVector3d(0.f,0.7f,0.7f));

  return true;
}

void create_flat_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const Eigen::MatrixXd& P)
{
  Eigen::RowVector3d plane_normal, plane_point;
  igl::fit_plane(P, plane_normal, plane_point);
  
  Eigen::RowVector3d bb_min = P.colwise().minCoeff();
  Eigen::RowVector3d bb_max = P.colwise().maxCoeff();
  Eigen::RowVector3d bb_d = (bb_max - bb_min).cwiseAbs();
  
  
  const Eigen::Vector3d P_centr = P.colwise().mean();
  const Eigen::Vector3d m_centr = (bb_max + bb_min) * 0.5;
  const Eigen::Affine3d t(Eigen::Translation3d(P_centr - m_centr));
  const Eigen::Affine3d r(getBasis(plane_normal));
  
  const Eigen::Matrix4d transform = t.matrix() * r.matrix();
  
  std::sort(bb_d.data(),bb_d.data()+bb_d.size());
  
  const double stepl = 0.01;
  
  const int y_steps = bb_d(0)/stepl;
  const int x_steps = bb_d(1)/stepl;
  
  
  V.resize(y_steps*x_steps, 3);
  F.resize(y_steps*x_steps*2, 3);

#pragma omp parallel for
  for (int y_step = 0; y_step < y_steps; ++y_step)
  {
    for (int x_step = 0; x_step < x_steps; ++x_step)
    {
      Eigen::RowVector4d v; v << (bb_min + Eigen::RowVector3d(x_step*stepl,y_step*stepl,0)),1.0;
      V.row(x_step + y_step*x_steps) << (v * transform.transpose()).head<3>();
    }
  }
  
#pragma omp parallel for
  for (int y_step = 0; y_step < y_steps-1; ++y_step)
  {
    for (int x_step = 0; x_step < x_steps-1; ++x_step)
    {
      F.row(x_step*2 + y_step*x_steps*2) << x_step+y_step*x_steps,x_step+1+y_step*x_steps,x_step+(y_step+1)*x_steps;
      F.row(x_step*2 + y_step*x_steps*2 + 1) << x_step+1+(y_step+1)*x_steps,x_step+(y_step+1)*x_steps,x_step+1+y_step*x_steps;
    }
  }
}

int main(int argc, char *argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    if (argc != 2)
    {
      std::cout << "Usage: $0 <folder>" << std::endl;
      return EXIT_SUCCESS;
    }

    std::string folder(argv[1]);
    
    DataSet ds(folder);
    
    /* GLFW has a mechanism for associating a "user" pointer with a GLFWWindow,
     * I tried it with viewer.window but it gets reset somehow...
     * This is awful but works.
     */
    global_ds_ptr = &ds;
    
    Eigen::MatrixXd points;
    Eigen::Matrix4d world2cam;
    if (!ds.get_next_point_cloud(points, world2cam))
    {
      std::cerr << "Couldn't read data" << std::endl;
      return EXIT_FAILURE;
    }
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    create_flat_mesh(V, F, points);
    
    /*
    Eigen::MatrixXd points_to_interpolate = points;
    
    for (int i = 0; i < points.rows(); ++i)
    {
      points_to_interpolate.row(i)(2) = 0;
    }

    
    Eigen::MatrixXd normals(points_to_interpolate.rows(),3);
    for (int i = 0; i < normals.rows(); ++i)
    {
      normals.row(i) << 0.0,0.0,-1.0;
    }
    Eigen::MatrixXd bc = igl::embree::line_mesh_intersection(points_to_interpolate, normals, V, F);
    
    std::cout << points.row(0) << std::endl << V.row(0) << std::endl;
*/
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(points);
    viewer.data().point_size = 5;
    viewer.data().add_points(points, Eigen::RowVector3d(0.f,0.7f,0.7f));
    //viewer.data().set_mesh(V, F);

    viewer.launch();
}
