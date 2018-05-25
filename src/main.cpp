#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#include "igl/fit_plane.h"
#include "igl/mat_min.h"
#include "DataSet.hpp"
#include "DataROS.hpp"
#include "Mesh.hpp"
#include <Eigen/Geometry>

#include <GLFW/glfw3.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <iostream>
#include <fstream>
#include <sstream>

int viewer_mesh_level = 0;
Mesh<float> mesh;
igl::opengl::glfw::Viewer viewer;
Eigen::MatrixXd P, P2, current_target, C;
DataSet *ds;
DataROS *dr;

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
    Eigen::MatrixXf vertices;
    Eigen::MatrixXi faces;
    mesh.get_mesh(mesh_level, vertices, faces);
    viewer.data().set_mesh(vertices.cast<double>(), faces);
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
}

void reload_viewer_data(igl::opengl::glfw::Viewer &viewer, const Eigen::MatrixXd pc, const unsigned int mesh_level)
{    
   reload_viewer_data(viewer, pc, C, mesh_level);
}

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int /*modifier*/)
{
  if (key == '1')
  {
    mesh.solve(20);
    Eigen::MatrixXf vertices;
    Eigen::MatrixXi faces;
    mesh.get_mesh(viewer_mesh_level, vertices, faces);
    viewer.data().set_mesh(vertices.cast<double>(), faces);
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
  }
  
  if (key == '2')
  {
    
    C.resize(1, 3);
    C << 0.f,0.7f,0.7f;
    
    if (ds) {
      Eigen::Matrix4d t_camera;
      ds->get_next_point_cloud(current_target, C, t_camera);
    }else
      current_target = P2;
      
    Eigen::MatrixXf cf = current_target.cast<float>();
    mesh.set_target_point_cloud(cf);
    reload_viewer_data(viewer, current_target, C, viewer_mesh_level);
    
  }
  
 /* if (key == '3' && ds)
  {
    Eigen::Matrix4d t_camera;
    for (int i = 0; i < 5; ++i)
    {
      if (!ds->get_next_point_cloud(current_target, C, t_camera))
        break;
        
      mesh.set_target_point_cloud(current_target);
      mesh.solve(20);
    }
    reload_viewer_data(viewer, current_target, C, viewer_mesh_level);
    
  }*/
  
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

  if (key == '4')
  {
    ds->clip_point_clouds();
  }
  
  
  return true;
}

void receive_viewer_data(const Eigen::MatrixXd points, const Eigen::MatrixXd colors) {
    P = points;
    C = colors;
}

int main(int argc, char* argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);
    
    std::thread t1;

    C.resize(1,3);
    C << 0.f,0.7f,0.7f;

    
    if (argc > 2)
    {
      std::cout << "Usage: $0" << std::endl;
      return EXIT_SUCCESS;
    } else if (argc > 1 && std::string(argv[1]) == "live") {
      dr = new DataROS(argc, argv);
      t1 = std::thread([] { dr->subscribe(&receive_viewer_data); });

      viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & v)->bool
      {
        v.data().clear();
        v.data().point_size=5;
        v.data().add_points(P, C);
        return false;
      };
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

    viewer.callback_key_down = callback_key_down;
    
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
      ImGui::Text("Iterate: \"1\"");
      ImGui::Text("Next point cloud: \"2\"");
      ImGui::Text("Current mesh level: %d", viewer_mesh_level);
    };
    
    // Set Viewer to tight draw loop
    viewer.core.is_animating = true;

    if (current_target.rows() > 0) {
      mesh.align_to_point_cloud(P.cast<float>().eval());
      mesh.set_target_point_cloud(current_target.cast<float>().eval());
      reload_viewer_data(viewer, current_target, C, viewer_mesh_level);
      viewer.core.align_camera_center(current_target);
    }

    viewer.launch();

    // cleanup
    mesh.cleanup();

    if (argc > 1 && std::string(argv[1]) == "live") {
      t1.join();
    }

    return 0;
}
