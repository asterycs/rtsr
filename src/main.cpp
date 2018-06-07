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

#include "igl/readPLY.h"

int viewer_mesh_level = 0;
Mesh<double> mesh;
Eigen::MatrixXd P, P2, current_target;
DataSet *ds;


Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd N;
Eigen::MatrixXd UV;
Eigen::MatrixXd VC;


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

void reload_viewer_data(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd pc, const Eigen::MatrixXd C, const unsigned int mesh_level)
{   
    viewer.data().clear();
    viewer.data().point_size = 5;
    viewer.data().add_points(pc, C);
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    ColorData colorData;

    mesh.get_mesh(mesh_level, vertices, faces, colorData);
    viewer.data().set_mesh(vertices, faces);
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
    viewer.data().set_texture(colorData.texture_red, colorData.texture_green, colorData.texture_blue);
    viewer.data().set_uv(colorData.UV);
    viewer.data().show_texture = true;
}

void reload_viewer_data(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd pc, const unsigned int mesh_level)
{
    Eigen::MatrixXd C(1, 3);
    C << 0.f,0.7f,0.7f;
    
   reload_viewer_data(viewer, pc, C, mesh_level);
}

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int /*modifier*/)
{
  if (key == '1')
  {
    ColorData c;
    mesh.solve(100);
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    mesh.get_mesh(viewer_mesh_level, vertices, faces, c);
    viewer.data().set_mesh(vertices, faces);
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
  }
  
  if (key == '2')
  {
    
    Eigen::MatrixXd C(1, 3);
    C << 0.f,0.7f,0.7f;
    
    if (ds) {
      Eigen::Matrix4d t_camera;
      ds->get_next_point_cloud(current_target, C, t_camera);
    }else
      current_target = P2;
      
    mesh.set_target_point_cloud(current_target);
    reload_viewer_data(viewer, current_target, C, viewer_mesh_level);
    
  }
  
  if (key == '0')
  {
    viewer_mesh_level = viewer_mesh_level >= MESH_LEVELS - 1 ? viewer_mesh_level : viewer_mesh_level + 1;
    reload_viewer_data(viewer, current_target, viewer_mesh_level);
  }
    
  if (key == '-')
  {
    viewer_mesh_level = viewer_mesh_level < 1 ? viewer_mesh_level : viewer_mesh_level - 1;
    reload_viewer_data(viewer, current_target, viewer_mesh_level); 
  }
  
  if (key == 'D')
  {
    mesh.debug(viewer);
    std::cout << "debug" << std::endl;
  }
  
  
  return true;
}

int main(int argc, char* argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    /*Eigen::MatrixXd C(1, 3);
    // C << 0.f,0.7f,0.7f;
        
    if (argc > 2)
    {
      std::cout << "Usage: $0" << std::endl;
      return EXIT_SUCCESS;
    } else if (argc > 1) {
      std::string folder(argv[1]);
      ds = new DataSet(folder);
      Eigen::Matrix4d t_camera;
      ds->get_next_point_cloud(P, C, t_camera);
      current_target = P;
    } else {
      Eigen::VectorXi id1, id2;
      generate_example_point_cloud2(P, id1, id2);
      split_point_cloud(P, current_target, P2, id1, id2);
    }
    
    mesh.align_to_point_cloud(P); // Resets the mesh everytime it is called
    mesh.set_target_point_cloud(current_target, C);
    igl::opengl::glfw::Viewer viewer;    
    viewer.callback_key_down = callback_key_down;

    reload_viewer_data(viewer, current_target, C, viewer_mesh_level);
    viewer.core.align_camera_center(current_target);

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
      ImGui::Text("Iterate: \"1\"");
      ImGui::Text("Next point cloud: \"2\"");
      ImGui::Text("Current mesh level: %d", viewer_mesh_level);
    };

    viewer.launch();*/


    //************************************************MINION SAMPLE*********************************************
    // Load a mesh in OFF format
    std::vector<std::vector<double > > VList;
    std::vector<std::vector<int > > FList;
    std::vector<std::vector<double > > NList;
    std::vector<std::vector<double > > CList;

    const std::string str = argv[1];
    igl::readOFF(str, VList, FList, NList, CList);

    V.resize(VList.size(), 3);
    VC.resize(CList.size(), 3);

    for(int i=0; i<V.rows(); i++)
    {
      V(i, 0) = VList[i][0];
      V(i, 1) = VList[i][1];
      V(i, 2) = VList[i][2];

      VC(i, 0) = CList[i][0];
      VC(i, 1) = CList[i][1];
      VC(i, 2) = CList[i][2];
    }
  

    mesh.align_to_point_cloud(V); 
    mesh.set_target_point_cloud(V, VC);

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;

    reload_viewer_data(viewer, V, VC, viewer_mesh_level);
    viewer.core.align_camera_center(V);

    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.callback_key_down = callback_key_down;
    viewer.plugins.push_back(&menu);

    viewer.data().point_size = 2;
    viewer.data().add_points(V, VC);

    menu.callback_draw_viewer_menu = [&]()
    {
      ImGui::Text("Iterate: \"1\"");
      ImGui::Text("Next point cloud: \"2\"");
      ImGui::Text("Current mesh level: %d", viewer_mesh_level);
    };
    // viewer.core.align_camera_center(V,F);
    viewer.launch();

}
