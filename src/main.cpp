#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "DataSet.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) {
    std::cout << "Keyboard callback!" << std::endl;

    return true;
}

void create_flat_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, const Eigen::MatrixXd& P)
{
  Eigen::RowVector3d bb_min = P.colwise().minCoeff();
  Eigen::RowVector3d bb_max = P.colwise().maxCoeff();
  
  const double stepl = 1;
  
  const int y_steps = (bb_max(1) - bb_min(1))/stepl;
  const int x_steps = (bb_max(0) - bb_min(0))/stepl;
  
  V.resize(y_steps*x_steps, 3);
  F.resize(y_steps*x_steps*2, 3);

#pragma omp parallel for
  for (int y_step = 0; y_step < y_steps; ++y_step)
  {
    for (int x_step = 0; x_step < x_steps; ++x_step)
    {
      V.row(x_step + y_step*x_steps) << x_step*stepl,y_step*stepl,0;
    }
  }
  
#pragma omp parallel for
  for (int y_step = 0; y_step < y_steps-1; ++y_step)
  {
    for (int x_step = 0; x_step < x_steps-1; ++x_step)
    {
      //std::cout << x_step+y_step*x_steps << " " << x_step+1+y_step*x_steps << " " << x_step+(y_step+1)*x_steps << std::endl;
      //std::cout << V.row(x_step+y_step*x_steps) << std::endl << V.row(x_step+1+y_step*x_steps) << std::endl << V.row(x_step+(y_step+1)*x_steps) << std::endl;
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

    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(points);
    viewer.data().point_size = 5;
    //viewer.data().add_points(points, Eigen::RowVector3d(0.7,0.7f,0.f));
    //viewer.data().add_points(V, Eigen::RowVector3d(0.7,0.7f,0.f));
    
    viewer.data().set_mesh(V, F);

    viewer.launch();
}
