#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOFF.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Eigen/Geometry"

#include <iostream>
#include <fstream>
#include <sstream>
#include <experimental/filesystem>

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers) {
    std::cout << "Keyboard callback!" << std::endl;

    return true;
}

std::string& trim_left(std::string& str, const char* t = " \t\n\r\f\v")
{
    str.erase(0, str.find_first_not_of(t));
    return str;
}

bool readWorld2Camera(std::vector<Eigen::Matrix4d>& world2cameras, const std::string& folder)
{
    std::ifstream camera_ref_file(folder + "/groundtruth.txt");
    
    if (!camera_ref_file.is_open())
    {
      std::cerr << "Couldn't open groundtruth.txt" << std::endl;
      return false;
    }
    
    world2cameras.clear();
    std::string line;
    
    while (std::getline(camera_ref_file, line))
    {
      if (line.empty())
        break;
        
      if (trim_left(line)[0] == '#')
        continue;
      
      std::stringstream line_stream(line);
            
      double time, tx, ty, tz, qi, qj, qk, ql;
      
      line_stream >> time >> tx >> ty >> tz >> qi >> qj >> qk >> ql;
      
      Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(tx, ty, tz)));
      Eigen::Affine3d r(Eigen::Quaterniond(qi, qj, qk, ql));
      
      Eigen::Matrix4d cam = t.matrix() * r.matrix();
      
      world2cameras.push_back(cam);
    }
    
    return true;
}

bool readDepthMaps(const std::vector<Eigen::Matrix4d>& word2camera, std::vector<Eigen::MatrixXd>& points, const std::string& folder)
{
  int img_cntr = 0;
  points.clear();  
  
  for (auto& p : std::experimental::filesystem::directory_iterator(folder + "/depth/"))
  {
    int width, height, bpp;
    unsigned char* png = stbi_load(p.path().string().c_str(), &width, &height, &bpp, 1);
    
    points.push_back(Eigen::MatrixXd(width*height, 3));
    
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        points.back().row(x + y*width) << x,y,png[(x+y*width)*bpp];
      }
    }
    
    stbi_image_free(png);
    ++img_cntr;
    
    break;
  }
  
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
    
    std::vector<Eigen::Matrix4d> world2camera;
    
    if (!readWorld2Camera(world2camera, folder))
    {
      std::cerr << "Douldn't open \"groundtruth.txt\"" << std::endl;
      return EXIT_FAILURE;
    }
    
    
    std::vector<Eigen::MatrixXd> points;
    
    if (!readDepthMaps(world2camera, points, folder))
    {
      std::cerr << "Couldn't open depth data" << std::endl;
      return EXIT_FAILURE;
    }

    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = callback_key_down;
    
    viewer.data().clear();
    viewer.core.align_camera_center(points[0]);
    viewer.data().point_size = 5;
    viewer.data().add_points(points[0], Eigen::RowVector3d(0.7,0.7f,0.f));

    viewer.launch();
}
