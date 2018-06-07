#include "DataSet.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Eigen/Geometry"

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

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

void pixel_to_camera_coord(const int x, const int y, const int z, Eigen::Vector3d& camCoord)
{ 
  const double fx = 517.3; // Focal length
  const double fy = 516.5; // Focal length
  const double cx = 318.6; // optical centre
  const double cy = 255.3; // optical centre
  
  const double Z = static_cast<double>(z)/5000.0;
  const double X = (static_cast<double>(x) - cx)*Z/fx;
  const double Y = (static_cast<double>(y) - cy)*Z/fy;
  
  camCoord << X,Y,Z;
}

std::string trim_left(const std::string& str, const char* t = " \t\n\r\f\v")
{
  std::string res = str;
  res.erase(0, res.find_first_not_of(t));
  return res;
}

DataSet::DataSet(const std::string& folder) : next_file_idx(0), folder_path(folder), camera_ref_file_name(folder+"groundtruth.txt"), rgb_ref_file_name(folder+"rgb.txt")
{
  boost::filesystem::directory_iterator depth_it(folder + "depth/");
  
  for (auto& it : depth_it)
    depth_files.push_back(it.path().string());
    
  std::sort(depth_files.begin(), depth_files.end());
      
  operational = true;
}

DataSet::~DataSet()
{

}


bool DataSet::get_next_point_cloud(Eigen::MatrixXd& points, Eigen::MatrixXd &colors, Eigen::Matrix4d& t_camera)
{
  if (!operational)
    return false;
    
  std::string time_stamp = strip_file_suffix(get_file_name(depth_files[next_file_idx]));
  double time_stamp_d = std::stod(time_stamp);
 
  if (!get_next_camera(t_camera, time_stamp_d))
    return false;
    
  const std::string rgb_filename = get_next_rgb(time_stamp_d);
  if (rgb_filename == "")
    return false;
  
  int rgb_width, rgb_height, rgb_bpp;  
  unsigned char* rgb = stbi_load((folder_path + rgb_filename).c_str(), &rgb_width, &rgb_height, &rgb_bpp, 3);
  
  if (static_cast<unsigned int>(next_file_idx) < depth_files.size())
  {    
    int width, height, bpp;
    unsigned short* png = stbi_load_16(depth_files[next_file_idx].c_str(), &width, &height, &bpp, 1);
    
    points.resize(width*height, 3);
    colors.resize(width*height, 3);
    
    int rowcntr = 0;
//#pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        if (png[(x+y*width)*bpp] == 0)
          continue;
          
        //if (x % 8 != 0 || y % 8 != 0)
        //  continue;
          
        Eigen::Vector3d camCoord;
        pixel_to_camera_coord(x, y, png[(x+y*width)/bpp], camCoord);
        
        Eigen::Vector4d camera_point; camera_point << camCoord,1.0;
        camera_point = t_camera * camera_point;
        Eigen::RowVector4d world_point = camera_point.transpose();
        unsigned char* color = rgb + (x+y*rgb_width)*rgb_bpp;
        
//#pragma omp critical
        {
          points.row(rowcntr) << world_point[0], world_point[2], world_point[1];
          colors.row(rowcntr) << (float)color[0]/255, (float)color[1]/255, (float)color[2]/255;
          ++rowcntr;
        }
      }
    }
    
    points.conservativeResize(rowcntr, Eigen::NoChange);
    colors.conservativeResize(rowcntr, Eigen::NoChange);

    stbi_image_free(png);
    stbi_image_free(rgb);
    next_file_idx++;
  }
  
  return true;
}

struct CameraEntry
{
  double time, tx, ty, tz, qi, qj, qk, ql;
  
  CameraEntry() : time(0), tx(0), ty(0), tz(0), qi(0), qj(0), qk(0), ql(0) {};
};

bool DataSet::get_next_camera(Eigen::Matrix4d& cam, const double timestamp)
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
      const double ti = (timestamp - previous.time) / (current.time - previous.time);
      const Eigen::Quaterniond qa(previous.ql,previous.qi,previous.qj,previous.qk),
                               qb(current.ql,current.qi,current.qj,current.qk);
      const Eigen::Quaterniond q_interp = qa.slerp(ti, qb);
      
      const Eigen::Vector3d ta(previous.tx, previous.ty, previous.tz),
                            tb(current.tx, current.ty, current.tz);
      const Eigen::Translation3d t_interp = Eigen::Translation3d(ta + ti * (tb - ta));
      
      const Eigen::Affine3d t(t_interp);
      const Eigen::Affine3d r(q_interp);
      
      cam << t.matrix() * r.matrix();
      abort = true;
    }else{
      previousDT = currentDT;
      previous = current;
    }
  }
    
  return true;
}

std::string DataSet::get_next_rgb(const double timestamp)
{
  std::fstream camera_ref_file(rgb_ref_file_name);
  double previousDT = std::numeric_limits<double>::max();
  std::string previous_filename;
  
  std::string line;
  while (std::getline(camera_ref_file >> std::ws, line) && !line.empty())
  {
    if (trim_left(line)[0] == '#')
      continue;
    
    std::stringstream ss(line);
    std::vector<std::string> result;
    while( ss.good() )
    {
        std::string substr;
        std::getline( ss, substr, ' ' );
        result.push_back( substr );
    }
    
    double t(atof(result[0].c_str()));
    std::string filename(result[1]);
    
    double currentDT = std::abs(t - timestamp);
    if (currentDT > previousDT) // Previous was closer
    {
      return previous_filename;
    }
    previous_filename = filename;
    previousDT = currentDT;
  }
  
  return "";
}
