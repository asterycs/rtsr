#include "DataSet.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Eigen/Geometry"

#include <iostream>

std::string& trim_left(std::string& str, const char* t = " \t\n\r\f\v")
{
    str.erase(0, str.find_first_not_of(t));
    return str;
}

DataSet::DataSet(const std::string& folder) : depth_it(folder + "/depth/"), camera_ref_file(folder + "/groundtruth.txt")
{
  if (!camera_ref_file.is_open())
  {
    std::cerr << "Couldn't open groundtruth.txt" << std::endl;
    operational = false;
  }
  
  operational = true;
}

DataSet::~DataSet()
{
  camera_ref_file.close();
}


bool DataSet::get_next_point_cloud(Eigen::MatrixXd& points, Eigen::Matrix4d& world2camera)
{
  if (!operational)
    return false;
  
  if (depth_it != std::experimental::filesystem::directory_iterator())
  {
    auto& p = depth_it;
    int width, height, bpp;
    unsigned char* png = stbi_load(p->path().string().c_str(), &width, &height, &bpp, 1);
    
    points.resize(width*height, 3);
    
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        points.row(x + y*width) << x,y,png[(x+y*width)*bpp];
      }
    }
    
    stbi_image_free(png);
    ++depth_it;
  }
  
  return true;
}


bool DataSet::getNextCamera(Eigen::Matrix4d& cam)
{
  if (!operational)
    return false;
    
  std::string line;
  
  while (std::getline(camera_ref_file, line) && trim_left(line)[0] == '#');
  
  if (line.empty())
    return false;
  
  std::stringstream line_stream(line);
        
  double time, tx, ty, tz, qi, qj, qk, ql;
  
  line_stream >> time >> tx >> ty >> tz >> qi >> qj >> qk >> ql;
  
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(tx, ty, tz)));
  Eigen::Affine3d r(Eigen::Quaterniond(qi, qj, qk, ql));
  
  cam = t.matrix() * r.matrix();
  
  return true;
}


