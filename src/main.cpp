#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "DataSet.hpp"
#include "Mesh.hpp"
#include <Eigen/Geometry>

#include <GLFW/glfw3.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <iostream>
#include <fstream>
#include <sstream>

Mesh<double> mesh;
Eigen::MatrixXd P, P1, P2, C;
DataSet *ds;

template <typename T>
void split_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P1, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P2, const Eigen::Matrix<int, Eigen::Dynamic, 1>& id1, Eigen::Matrix<int, Eigen::Dynamic, 1>& id2)
{
  P1.resize(id1.rows(), 3);
  P2.resize(id2.rows(), 3);
  
  for (int i = 0; i < id1.rows(); ++i)
    P1.row(i) = P.row(id1(i));
  
  for (int i = 0; i < id2.rows(); ++i)
    P2.row(i) = P.row(id2(i));
}

template <typename T>
void generate_example_point_cloud(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, Eigen::Matrix<int, Eigen::Dynamic, 1>& id1, Eigen::Matrix<int, Eigen::Dynamic, 1>& id2)
{
  const int size = 120;
  P.resize(size*size,3);
  id1.resize(size*int(size / 2));
  id2.resize(size*int((size+1) / 2));
  
  int cntr1 = 0, cntr2 = 0;
  for (int x = 0; x < size; ++x)
  {
    const int multipl = x / 30 + 1;
    for (int y = 0; y < size; ++y)
    {
      T height;
      
      if (int(y / (size / (5*multipl))) % 2)
        height = 15;
      else
        height = 0;
        
      Eigen::Matrix<T, 1, 3> row(T(x), height, T(y));
      P.row(x + y*size) = row;
      
      if (x < size / 2)
      {
        id1(cntr1++) = x + y*size;
      }
      else
      {
        id2(cntr2++) = x + y*size;
      }
        
    }
  }
  
}

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int /*modifier*/)
{
  if (key == '1')
  {
    mesh.iterate();
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
  }
  
  if (key == '2')
  {
    
    C.resize(1, 3);
    C << 0.f,0.7f,0.7f;
    
    if (ds) {
      Eigen::Matrix4d t_camera;
      ds->get_next_point_cloud(P2, C, t_camera);
    }
    viewer.data().clear();
    mesh.set_target_point_cloud(P2);
    viewer.data().add_points(P2, C);
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
    
  }
  
  return true;
}

int main(int argc, char* argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    C.resize(1,3);
    C << 0.f,0.7f,0.7f;
    
    if (argc > 2)
    {
      std::cout << "Usage: $0" << std::endl;
      return EXIT_SUCCESS;
    } else if (argc > 1) {
      std::string folder(argv[1]);
      ds = new DataSet(folder);
      Eigen::Matrix4d t_camera;
      ds->get_next_point_cloud(P, C, t_camera);
      P1 = P;
    } else {
      Eigen::VectorXi id1, id2;
      generate_example_point_cloud(P, id1, id2);
      split_point_cloud(P, P1, P2, id1, id2);
    }
    
    mesh.align_to_point_cloud(P); // Resets the mesh everytime it is called
    mesh.set_target_point_cloud(P1);
    
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(P);
    viewer.data().point_size = 5;
    viewer.data().add_points(P1, C);
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
      ImGui::Text("Iterate: \"1\"");
      ImGui::Text("Next point cloud: \"2\"");
    };
    
    viewer.launch();
}
