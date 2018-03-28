#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "igl/embree/line_mesh_intersection.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "DataSet.hpp"
#include "Mesh.hpp"
#include <Eigen/Geometry>

#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>

Mesh<double> mesh;

template <typename T>
void generate_example_point_cloud(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  const int size = 60;
  P.resize(size*size,3);
  
  for (int x = 0; x < size; ++x)
  {
    for (int y = 0; y < size; ++y)
    {
      Eigen::Matrix<T, 1, 3> row(T(x), 4., T(y));
      P.row(x + y*size) = row;
    }
  }
  
  const int b = 2;
  const int s = 7;
  const int step = size / (b+1);
  int cntr = 0;

  for (int bx = step; bx < size; bx += step)
  {
    for (int by = step; by < size; by += step)
    {
      for (int x = bx-s; x < bx+s; ++x)
      {
        for (int y = by-s; y < by+s; ++y)
        {
          Eigen::Matrix<T, 1, 3> row(T(x), cntr * 4., T(y));
          P.row(x + y*size) = row;
        }
      }
      
      ++cntr;
    }
  }
}

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers)
{
  mesh.iterate();
  viewer.data().set_mesh(mesh.vertices(), mesh.faces());
  viewer.data().compute_normals();
  viewer.data().invert_normals = true;
  
  return true;
}

int main(int argc, char *argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    if (argc != 1)
    {
      std::cout << "Usage: $0" << std::endl;
      return EXIT_SUCCESS;
    }
    
    Eigen::MatrixXd P;
    generate_example_point_cloud(P);
    
    mesh.align_to_point_cloud(P); // Resets the mesh everytime it is called
    mesh.set_target_point_cloud(P); // Use this to solve for the height field
    
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(P);
    viewer.data().point_size = 5;
    viewer.data().add_points(P, Eigen::RowVector3d(0.f,0.7f,0.7f));
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
    
    viewer.launch();
}
