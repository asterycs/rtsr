#include "DataSet.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Eigen/Geometry"

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

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

DataSet::DataSet(const std::string& folder) : next_file_idx(0), folder_path(folder), rgb_ref_file_name(folder+"rgb.txt")
{
  boost::filesystem::directory_iterator depth_it(folder + "depth/");
  
  const std::string groundtruth_ref_file_name(folder+"groundtruth.txt");
  const std::string camera_ref_file_name(folder+"CameraTrajectory.txt");
  
  camera_ref_file.open(groundtruth_ref_file_name.c_str());
  if (!camera_ref_file.is_open()) {
    std::cout << "Couldn't open groundtruth.txt - trying CameraTrajectory.txt" << std::endl;
    std::cout << camera_ref_file_name << std::endl;
    camera_ref_file.open(camera_ref_file_name.c_str());
  }
  if (!camera_ref_file.is_open()) {
    std::cerr << "Couldn't open groundtruth.txt" << std::endl;
    return;
  }

  for (auto& it : depth_it)
    depth_files.push_back(it.path().string());
    
  std::sort(depth_files.begin(), depth_files.end());

  rOffset = Eigen::Matrix4d::Identity();
  clip = false;
  operational = true;

  Eigen::MatrixXd P, C;
  Eigen::Matrix4d t_camera;
  get_next_point_cloud(P, C, t_camera);
  Eigen::Affine3f r = findPlaneRotation(P);
  rOffset = r.matrix().cast<double>();
  clip = true;
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
  t_camera = rOffset * t_camera;
    
  std::string rgb_filename = get_next_rgb(time_stamp_d);
  if (rgb_filename == "")
    return false;
  
  int rgb_width, rgb_height, rgb_bpp;
  std::string rgb_path = folder_path + rgb_filename;
  unsigned char* rgb = stbi_load(rgb_path.c_str(), &rgb_width, &rgb_height, &rgb_bpp, 3);
  
  if (static_cast<unsigned int>(next_file_idx) < depth_files.size())
  {    
    int width, height, bpp;
    unsigned short* png = stbi_load_16(depth_files[next_file_idx].c_str(), &width, &height, &bpp, 1);
    
    points.resize(width*height, 3);
    colors.resize(width*height, 3);
    
    int rowcntr = 0;
#pragma omp parallel for
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
        
        if (clip && world_point[1] < 0.05) continue; // so not the shadowy side?

#pragma omp critical
        {
          points.row(rowcntr) << world_point[0], world_point[1], world_point[2];
          colors.row(rowcntr) << (float)color[0]/255, (float)color[1]/255, (float)color[2]/255;
          ++rowcntr;
        }
      }
    }
    
    points.conservativeResize(rowcntr, Eigen::NoChange);
    stbi_image_free(png);
    stbi_image_free(rgb);
    next_file_idx++;
  }
  
  return true;
}

void DataSet::clip_point_clouds() {
  clip = !clip;
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
    
  std::string line;
  bool abort = false;
  
  struct CameraEntry previous;
  previous.time = 0;
  
  camera_ref_file.seekg(0, std::ios::beg);

  while (!abort)
  {
    while (std::getline(camera_ref_file, line) && trim_left(line)[0] == '#');
    
    if (line.empty())
      return false;
    
    std::stringstream line_stream(line);
    
    struct CameraEntry current;
    
    line_stream >> current.time >> current.tx >> current.ty >> current.tz >> current.qi >> current.qj >> current.qk >> current.ql;
    if (current.time > timestamp && previous.time > 0) // Previous was closer
    {
      // std::cout << (previous.time - 1527117859) << " " << (timestamp - 1527117859) << " " << (current.time - 1527117859) << std::endl;
      const double ti = (timestamp - previous.time) / (current.time - previous.time);
      const Eigen::Quaterniond qa(previous.ql,previous.qi,previous.qj,previous.qk),
                               qb(current.ql,current.qi,current.qj,current.qk);
      const Eigen::Quaterniond q_interp = qa.slerp(ti, qb);

      // std::cout << (timestamp-1527117859) << " " << ti << " " << q_interp.x() << " " << q_interp.y() << " " << q_interp.z() << " " << q_interp.w() << std::endl;
      
      const Eigen::Vector3d ta(previous.tx, previous.ty, previous.tz),
                            tb(current.tx, current.ty, current.tz);
      const Eigen::Translation3d t_interp = Eigen::Translation3d(ta + ti * (tb - ta));
      
      const Eigen::Affine3d t(t_interp);
      const Eigen::Affine3d r(q_interp);
      
      cam << t.matrix() * r.matrix();
      abort = true;
    }else{
      previous = current;
    }
  }
    
  return true;
}

std::string DataSet::get_next_rgb(const double timestamp)
{
  std::fstream rgb_ref_file(rgb_ref_file_name);
  double previousDT = std::numeric_limits<double>::max();
  std::string previous_filename;
  
  std::string line;
  while (std::getline(rgb_ref_file >> std::ws, line) && !line.empty())
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

void DataROS::subscribe(void (*getData)(const Eigen::MatrixXd points, const Eigen::MatrixXd colors)) {  
  subscriber = getData;

  if (subscriber != NULL && rowcntr > 0) {
    subscriber(points, colors);
  }
}


Eigen::Affine3f DataSet::findPlaneRotation(Eigen::MatrixXd& points) {
  // Use Ransac

  long int size = points.rows();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = size;
  cloud->height = 1;
  cloud->points.resize (cloud->width * cloud->height);

#pragma omp parallel for
  for (int i = 0; i < size; ++i)
  {
    cloud->points[i].x = (float)points(i,0);
    cloud->points[i].y = (float)points(i,1);
    cloud->points[i].z = (float)points(i,2);
  }

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.01);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return Eigen::Affine3f::Identity();
  }

  // std::cout << "Model coefficients: " << coefficients->values[0] << " " 
  //                                     << coefficients->values[1] << " "
  //                                     << coefficients->values[2] << " " 
  //                                     << coefficients->values[3] << std::endl;

  Eigen::Matrix<float, 1, 3> floor_plane_normal_vector, xy_plane_normal_vector;

  floor_plane_normal_vector[0] = coefficients->values[0];
  floor_plane_normal_vector[1] = coefficients->values[1];
  floor_plane_normal_vector[2] = coefficients->values[2];

  if (floor_plane_normal_vector[1] > 0) floor_plane_normal_vector = -floor_plane_normal_vector;

  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 1.0;
  xy_plane_normal_vector[2] = 0.0;

  Eigen::Vector3f mu (0.0, 0.0, 0.0);

  for (uint i=0;i < inliers->indices.size();i++){

    // Get Point
    pcl::PointXYZ pt = cloud->points[inliers->indices[i]];

    mu[0] += pt.x;
    mu[1] += pt.y;
    mu[2] += pt.z;
  }

  mu /= (float)inliers->indices.size ();

  Eigen::Vector3f rotation_vector = xy_plane_normal_vector.cross(floor_plane_normal_vector);
  rotation_vector.normalize();
  float theta = -atan2f(rotation_vector.norm(), xy_plane_normal_vector.dot(floor_plane_normal_vector));

  Eigen::Affine3f transform_1(Eigen::Translation3f(-mu));

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  transform_2.rotate(Eigen::AngleAxisf (theta, rotation_vector));

  return transform_2 * transform_1;
}
