#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "igl/embree/line_mesh_intersection.h"
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
Eigen::MatrixXd P, P1, P2;

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
void generate_example_point_cloud1(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  const int size = 120;
  P.resize(size*size,3);
  
  for (int x = 0; x < size; ++x)
  {
    for (int y = 0; y < size; ++y)
    {
      Eigen::Matrix<T, 1, 3> row(T(x), 4., T(y));
      P.row(x + y*size) = row;
    }
  }
  
  const int b = 3;
  const int s = 5;
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
          Eigen::Matrix<T, 1, 3> row(T(x), cntr * 1., T(y));
          P.row(x + y*size) = row;
        }
      }
      
      ++cntr;
    }
  }
}

template <typename T>
void generate_example_point_cloud2(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, Eigen::Matrix<int, Eigen::Dynamic, 1>& id1, Eigen::Matrix<int, Eigen::Dynamic, 1>& id2)
{
  const int size = 120;
  P.resize(size*size,3);
  id1.resize(size*size / 2);
  id2.resize(size*size / 2);
  
  int cntr1 = 0, cntr2 = 0;
  for (int x = 0; x < size; ++x)
  {
    for (int y = 0; y < size; ++y)
    {
      T height;
      
      if (y / (size / 5) % 2 == 0)
        height = 15;
      else
        height = 0;
        
      Eigen::Matrix<T, 1, 3> row(T(x), height, T(y));
      P.row(x + y*size) = row;
      
      if (x < size / 2)
        id1(cntr1++) = x + y*size;
      else
        id2(cntr2++) = x + y*size;
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
    viewer.data().clear();
    mesh.set_target_point_cloud(P2);
    viewer.data().add_points(P2, Eigen::RowVector3d(0.f,0.7f,0.7f));
    viewer.data().set_mesh(mesh.vertices(), mesh.faces());
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
  }
  
  return true;
}

int main(int argc, char **/*argv*/) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    if (argc != 1)
    {
      std::cout << "Usage: $0" << std::endl;
      return EXIT_SUCCESS;
    }
    
    Eigen::VectorXi id1, id2;
    generate_example_point_cloud2(P, id1, id2);
    split_point_cloud(P, P1, P2, id1, id2);
    
    mesh.align_to_point_cloud(P); // Resets the mesh everytime it is called
    mesh.set_target_point_cloud(P1);
    
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(P);
    viewer.data().point_size = 5;
    viewer.data().add_points(P1, Eigen::RowVector3d(0.f,0.7f,0.7f));
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
