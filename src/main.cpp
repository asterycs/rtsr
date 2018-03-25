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

DataSet* global_ds_ptr; // Dirty dirty...

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) {
    
  Eigen::MatrixXd points;
  Eigen::Matrix4d world2cam;
  
  if (!global_ds_ptr->get_next_point_cloud(points, world2cam))
  {
    std::cerr << "Couldn't read data" << std::endl;
    return false;
  }

  viewer.data().add_points(points, Eigen::RowVector3d(0.f,0.7f,0.7f));

  return true;
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
    
    Mesh<double> mesh;
    mesh.align_to_point_cloud(points); // Resets the mesh everytime it is called
    //mesh.solve(points); // Use this to solve for the height field
    
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(points);
    viewer.data().point_size = 5;
    viewer.data().add_points(points, Eigen::RowVector3d(0.f,0.7f,0.7f));
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());

    viewer.launch();
}
