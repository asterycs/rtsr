#include "DataSet.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Eigen/Geometry"

#include <iostream>
#include <iomanip>

std::string strip_file_suffix(const std::string& s)
{
  std::string::size_type idx = s.rfind('.');
  
  return std::string(s.substr(0, idx));
}

std::string get_file_name(const std::string& s)
{
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  size_t i = s.rfind(sep, s.length());
  if (i != std::string::npos)
  {
    return(s.substr(i+1, s.length() - i));
  }

  return("");
}

 
std::vector<double>::const_iterator closest(std::vector<double> const& vec, double value)
{
  auto const it = std::lower_bound(vec.begin(), vec.end(), value);

  return it;
}

void normalizedImageCoordinateFromPixelCoordinate(const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height, Eigen::Vector2d& coord)
{
  float pixelWidth  = 2.f / width;
  float pixelHeight = 2.f / height;

  coord = Eigen::Vector2d(-1.0 + 0.5 * pixelWidth + x * pixelWidth, -1.0 + 0.5 * pixelHeight + y * pixelHeight);
}

std::string& trim_left(std::string& str, const char* t = " \t\n\r\f\v")
{
    str.erase(0, str.find_first_not_of(t));
    return str;
}

DataSet::DataSet(const std::string& folder) : next_file_idx(0), camera_ref_file_name(folder+"groundtruth.txt")
{
  std::experimental::filesystem::directory_iterator depth_it(folder + "/depth/");
  
  for (auto& it : depth_it)
    depth_files.push_back(it.path().string());
    
  std::sort(depth_files.begin(), depth_files.end());
      
  operational = true;
}

DataSet::~DataSet()
{

}


bool DataSet::get_next_point_cloud(Eigen::MatrixXd& points, Eigen::Matrix4d& world2camera)
{
  if (!operational)
    return false;
    
  std::string time_stamp = strip_file_suffix(get_file_name(depth_files[next_file_idx]));
  double time_stamp_d = std::stod(time_stamp);
 
  if (!getNextCamera(world2camera, time_stamp_d))
    return false;
    
  const Eigen::Matrix4d world2camera_inv = world2camera;
  
  if (static_cast<unsigned int>(next_file_idx) < depth_files.size())
  {    
    int width, height, bpp;
    unsigned char* png = stbi_load(depth_files[next_file_idx].c_str(), &width, &height, &bpp, 1);
    
    points.resize(width*height, 3);
    
#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        if (png[(x+y*width)*bpp] == 0)
          continue;
          
        Eigen::Vector2d normCoord;
        normalizedImageCoordinateFromPixelCoordinate(x, y, width, height, normCoord);
        
        Eigen::RowVector4d camera_point(normCoord(0),normCoord(1),png[(x+y*width)/bpp]/100.0,1.0); // Scaling factor 5000 from data doc
        Eigen::RowVector4d world_point = world2camera_inv * camera_point.transpose();
        
        points.row(x + y*width) << world_point.head<3>();
      }
    }
    
    stbi_image_free(png);
    next_file_idx+=10;
  }
  
  return true;
}

struct CameraEntry
{
  double time, tx, ty, tz, qi, qj, qk, ql;
};

bool DataSet::getNextCamera(Eigen::Matrix4d& cam, const double timestamp)
{
  if (!operational)
    return false;
    
  std::fstream camera_ref_file(camera_ref_file_name);
  
  if (!camera_ref_file.is_open())
  {
    std::cerr << "Couldn't open groundtruth.txt" << std::endl;
    operational = false;
  }
    
  std::string line;
  bool abort = false;
  
  struct CameraEntry previous;
  double previousDT = std::numeric_limits<double>::max();
  
  while (!abort)
  {
    while (std::getline(camera_ref_file, line) && trim_left(line)[0] == '#');
    
    if (line.empty())
      return false;
    
    std::stringstream line_stream(line);
    
    struct CameraEntry current;
    
    line_stream >> current.time >> current.tx >> current.ty >> current.tz >> current.qi >> current.qj >> current.qk >> current.ql;
    double currentDT = std::abs(current.time - timestamp);
    if (currentDT > previousDT) // Previous was closer
    {
      Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(previous.tx, previous.ty, previous.tz)));
      Eigen::Affine3d r(Eigen::Quaterniond(previous.qi, previous.qj, previous.qk, previous.ql));

      cam << t.matrix() * r.matrix();
      abort = true;
    }else{
      previousDT = currentDT;
      previous = current;
    }
  }
    
  return true;
}


