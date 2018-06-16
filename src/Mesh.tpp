#include "Mesh.hpp"

#include "igl/barycentric_coordinates.h"
#include "Util.hpp"

#include <Eigen/QR>

#include <iostream>
#include <cstdlib>
#include <iostream>

template <typename T> using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using VecC3 = Eigen::Matrix<T, 3, 1>;
template <typename T> using VecC2 = Eigen::Matrix<T, 2, 1>;
template <typename T> using VecR3 = Eigen::Matrix<T, 1, 3>;

template <typename T>
Mesh<T>::Mesh()
{
#ifdef ENABLE_CUDA
  std::cout << " ----**** Creating mesh with CUDA solver ****----" << std::endl;
#else
  std::cout << " ----**** Creating mesh with CPU solver ****----" << std::endl;
#endif  

  color_counter =  Eigen::MatrixXi::Zero(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION);

  texture.red   = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Constant(TEXTURE_RESOLUTION,TEXTURE_RESOLUTION, static_cast<unsigned char>(115));
  texture.green = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Constant(TEXTURE_RESOLUTION,TEXTURE_RESOLUTION, static_cast<unsigned char>(115));
  texture.blue  = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Constant(TEXTURE_RESOLUTION,TEXTURE_RESOLUTION, static_cast<unsigned char>(115));
}

template <typename T>
Mesh<T>::~Mesh()
{
  std::ofstream residual_file;
  residual_file.open("error.csv");
  residual_file << "pc_idx,level,iteration,residuals\n";

  for (int pci = 0; pci < residuals.size(); ++pci)
  {
    for (int li = 0; li < residuals[pci].size(); ++li)
    {
      for (int it = 0; it < residuals[pci][li].size(); ++it)
        residual_file << pci << "," << li << "," << it << "," << residuals[pci][li][it] << std::endl;
    }
  }

  residual_file.close();
}

template <typename T>
void Mesh<T>::cleanup()
{
#ifdef ENABLE_CUDA
    cudaDeviceSynchronize();
#endif

    for (auto& m : JtJ)
        m.clear();

    for (auto& v : Jtz)
        v.clear();
}

inline int subdivided_side_length(const int level, const int base_resolution)
{
  return (base_resolution-1)*static_cast<int>(std::pow(2,level))+1;
}

template <typename T>
void upsample(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V, const Eigen::MatrixXi& F, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out)
{
    const int old_resolution = static_cast<int>(std::sqrt(static_cast<double>(V.rows())));
    const int resolution = subdivided_side_length(1, old_resolution);
    const int faces = (resolution - 1) * (resolution - 1) * 2;
    const int vertices = resolution * resolution;
    
    F_out = Eigen::MatrixXi::Zero(faces, 3);
    V_out = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(vertices, 3);
        
#pragma omp parallel for
    for (int y_step = 0; y_step < old_resolution-1; ++y_step)
    {
      for (int x_step = 0; x_step < old_resolution-1; ++x_step)
      {
        
        Eigen::Vector3i f_u = F.row(x_step*2 + y_step*(old_resolution-1)*2);
        Eigen::Vector3i f_l = F.row(x_step*2 + y_step*(old_resolution-1)*2 + 1);

        const VecC3<T> old_v0 = V.row(f_u(0));
        const VecC3<T> old_v1 = V.row(f_u(1));
        const VecC3<T> old_v2 = V.row(f_u(2));
        
        const VecC3<T> old_v3 = V.row(f_l(0));
        const VecC3<T> old_v4 = V.row(f_l(1));
        //const TvecC3 old_v5 = V.row(f_l(2));

        const VecC3<T> v0 = old_v0;
        const VecC3<T> v1 = (old_v0 + old_v1) * T(0.5);
        const VecC3<T> v2 = old_v1;
        const VecC3<T> v3 = (old_v0 + old_v2) * T(0.5);
        const VecC3<T> v4 = (old_v2 + old_v1) * T(0.5);
        const VecC3<T> v5 = (old_v3 + old_v1) * T(0.5);
        const VecC3<T> v6 = old_v2;
        const VecC3<T> v7 = (old_v3 + old_v4) * T(0.5);
        const VecC3<T> v8 = old_v3;

        const int new_v_xi = x_step * 2;
        const int new_v_yi = y_step * 2;
        
        int
          v0i = new_v_xi       + new_v_yi * resolution,
          v1i = new_v_xi + 1 + new_v_yi * resolution,
          v2i = new_v_xi + 2+ new_v_yi * resolution,
          v3i = new_v_xi + (new_v_yi+1) * resolution,
          v4i = new_v_xi + 1 + (new_v_yi+1) * resolution,
          v5i = new_v_xi + 2 + (new_v_yi+1) * resolution,
          v6i = new_v_xi       + (new_v_yi+2) * resolution,
          v7i = new_v_xi + 1 + (new_v_yi+2) * resolution,
          v8i = new_v_xi + 2 + (new_v_yi+2) * resolution;
        
        V_out.row(v0i) = v0;
        V_out.row(v1i) = v1;
        V_out.row(v3i) = v3;
        V_out.row(v4i) = v4;
        
        if (x_step == old_resolution-2)
        {
          V_out.row(v2i) = v2;
          V_out.row(v5i) = v5;
        }
        
        if (y_step == old_resolution-2)
        {
          V_out.row(v6i) = v6;
          V_out.row(v7i) = v7;
        }

        if (x_step == old_resolution - 2 && y_step == old_resolution-2)
          V_out.row(v8i) = v8;

        const int new_f_xi = x_step * 2;
        const int new_f_yi = y_step * 2;
        
        F_out.row(2*new_f_xi     + new_f_yi * (resolution-1) * 2) << v0i,v1i,v3i;
        F_out.row(2*new_f_xi + 1 + new_f_yi * (resolution-1) * 2) << v4i,v3i,v1i;
        F_out.row(2*new_f_xi + 2 + new_f_yi * (resolution-1) * 2) << v1i,v2i,v4i;
        F_out.row(2*new_f_xi + 3 + new_f_yi * (resolution-1) * 2) << v5i,v4i,v2i;
        F_out.row(2*new_f_xi +     (new_f_yi+1) * (resolution-1)*2) << v3i,v4i,v6i;
        F_out.row(2*new_f_xi + 1 + (new_f_yi+1) * (resolution-1)*2) << v7i,v6i,v4i;
        F_out.row(2*new_f_xi + 2 + (new_f_yi+1) * (resolution-1)*2) << v4i,v5i,v7i;
        F_out.row(2*new_f_xi + 3 + (new_f_yi+1) * (resolution-1)*2) << v8i,v7i,v5i;
      }
    }
}


template <typename T>
void upsample(const unsigned int levels, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V, const Eigen::MatrixXi& F, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out)
{
    if (levels < 1)
    {
      V_out = V;
      F_out = F;
      return;
    }
    
    Eigen::MatrixXi F_in = F;
    Mat<T> V_in = V;
    
    for (unsigned int i = 0; i < levels; ++i)
    {
      upsample(V_in, F_in, V_out, F_out);
      F_in = F_out;
      V_in = V_out;
    }
}

template <typename T>
template <typename Derived>
void Mesh<T>::align_to_point_cloud(const Eigen::MatrixBase<Derived>& P)
{    
  const VecR3<T> bb_min = P.colwise().minCoeff();
  const VecR3<T> bb_max = P.colwise().maxCoeff();
  const VecR3<T> bb_d = (bb_max - bb_min).cwiseAbs();
  
  JtJ.resize(MESH_LEVELS);
  Jtz.resize(MESH_LEVELS);
  V.resize(MESH_LEVELS);
  F.resize(MESH_LEVELS);

  for (int li = 0; li < MESH_LEVELS; ++li)
  {  
    const T scaling_factor(MESH_SCALING_FACTOR);
    const int resolution = subdivided_side_length(li, MESH_RESOLUTION);
    
    // Scaling matrix
    const Eigen::Transform<T, 3, Eigen::Affine> scaling(Eigen::Scaling(VecC3<T>(scaling_factor*bb_d(0)/T(resolution), 0.f,scaling_factor*bb_d(2)/T(resolution))));
    
    //const TvecR3 pc_mean = P.colwise().mean();
    // P_centr: mean of the point cloud
    VecR3<T> P_centr = bb_min + 0.5f*(bb_max - bb_min);
    P_centr(1) = 0.0;
    
    const Eigen::Transform<T, 3, Eigen::Affine> t(Eigen::Translation<T, 3>(P_centr.transpose()));
    
    transform = t.matrix();
      
    V[li].resize(resolution*resolution, 3);
    F[li].resize((resolution-1)*(resolution-1)*2, 3);
    JtJ[li].resize(resolution);
    Jtz[li].resize(resolution);

  #pragma omp parallel for
    for (int z_step = 0; z_step < resolution; ++z_step)
    {
      for (int x_step = 0; x_step < resolution; ++x_step)
      {
        Eigen::Matrix<T, 1, 4> v; v << VecR3<T>(T(x_step)-T(resolution-1)/2.f,T(0.0), T(z_step)-T(resolution-1)/2.f),T(1.0);
        VecR3<T> pos;
        
        if (li == 0) // Only transform the first layer. Subsequent layers only denot a difference between the first
          pos << (v * scaling.matrix().transpose() * transform.transpose()).template head<3>();
        else
          pos << (v * scaling.matrix().transpose()).template head<3>();
   
        V[li].row(x_step + z_step*resolution) << pos;
      }
    }
    
  #pragma omp parallel for
    for (int y_step = 0; y_step < resolution-1; ++y_step)
    {
      for (int x_step = 0; x_step < resolution-1; ++x_step)
      {
        F[li].row(x_step*2 + y_step*(resolution-1)*2)     << x_step+   y_step   *resolution,x_step+1+y_step*   resolution,x_step+(y_step+1)*resolution;
        F[li].row(x_step*2 + y_step*(resolution-1)*2 + 1) << x_step+1+(y_step+1)*resolution,x_step+ (y_step+1)*resolution,x_step+1+y_step*resolution;
      }
    }
        
    // Initialize Lh and rh with sensible values
    for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
    {
      JtJ[li].update_triangle(i, 1.f, 0.f);
      JtJ[li].update_triangle(i, 0.f, 1.f);
      JtJ[li].update_triangle(i, 0.f, 0.f);
    }

    for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
      Jtz[li].update_triangle(i, 0.34f, 0.33f, 0.f);
  }
}

template <typename T>
void Mesh<T>::get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out, ColorData& colordata) const
{
    const int mesh_resolution = JtJ[level].get_mesh_width();
    
    colordata.UV.resize(mesh_resolution*mesh_resolution, 2);
    const T tdx = T(1.0) / (mesh_resolution-1);
    for(int i = 0; i < mesh_resolution; i++)
    {
      for(int j = 0; j < mesh_resolution; j++)
      {
        const int index = j * mesh_resolution + i;
        colordata.UV(index, 0) = i*tdx;
        colordata.UV(index, 1) = j*tdx;
      }
    }
    
    colordata.texture.red = texture.red;
    colordata.texture.green = texture.green;
    colordata.texture.blue = texture.blue;
    
    get_mesh(level, V_out, F_out);
}

template <typename T>
void Mesh<T>::get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out) const
{
  if (level < 1 || level > MESH_LEVELS - 1)
  {
    V_out = V[0];
    F_out = F[0];
  }
  else
  {    
    upsample(level, V[0], F[0], V_out, F_out);
    
    for (unsigned int li = 1; li <= level; ++li)
    {
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V_upsampled;
      Eigen::MatrixXi F_upsampled;
      upsample(level-li, V[li], F[li], V_upsampled, F_upsampled);
      
       V_out.col(1) += V_upsampled.col(1);
    }
  }
}

// Compute barycentric coordinates based on the corner points of a triangle
template <typename T>
void barycentric(const VecC2<T>& p, const Eigen::Matrix<T, 2, 1>& a, const Eigen::Matrix<T, 2, 1>& b, const Eigen::Matrix<T, 2, 1>& c, T &u, T &v, T &w)
{
    Eigen::Matrix<T, 2, 1> v0 = b - a,
                                           v1 = c - a,
                                           v2 = p - a;
    T a00 = v0.dot(v0);
    T a01 = v0.dot(v1);
    T a11 = v1.dot(v1);
    T a20 = v2.dot(v0);
    T a21 = v2.dot(v1);
    T den = 1 / (a00 * a11 - a01 * a01);
    v =  (a11 * a20 - a01 * a21) * den;
    w = (a00 * a21 - a01 * a20) * den;
    u = T(1.0) - v - w;
}

// Project all points onto the 2D plane, we only need to project onto XZ plane as the algorithm only runs on height fields
// Also why we only need to project only when the input point cloud changes
template <typename T>
void Mesh<T>::project_points(const int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc) const
{
  bc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(current_target_point_cloud.rows(), 3);
  
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Vl;
  Eigen::MatrixXi Fl;
  get_mesh(level, Vl, Fl);
  
  const int resolution = subdivided_side_length(level, MESH_RESOLUTION);
  
  // Upper left triangle. Used to compute the index of the triangle that is hit
  const VecC2<T> V_ul(Vl.row(0)(0), Vl.row(0)(2));
  const VecC2<T> V_00(Vl.row(Fl.row(0)(0))(0), Vl.row(Fl.row(0)(0))(2));
  const VecC2<T> V_01(Vl.row(Fl.row(0)(1))(0), Vl.row(Fl.row(0)(1))(2));
  const VecC2<T> V_10(Vl.row(Fl.row(0)(2))(0), Vl.row(Fl.row(0)(2))(2));
  
  const double dx = (V_01- V_00).norm();
  const double dy = (V_10 - V_00).norm();
  
#pragma omp parallel for
  for (int pi = 0; pi < current_target_point_cloud.rows(); ++pi)
  {
    const VecC2<T> current_point(current_target_point_cloud.row(pi)(0), current_target_point_cloud.row(pi)(2));
    const VecC2<T> offset(current_point - V_ul);
    
    const int c = static_cast<int>(offset(0) / dx);
    const int r = static_cast<int>(offset(1) / dy);
    
    const int inner_size = resolution - 1;

    // Indices are outside of mesh borders => a triangle cannot be hit
    if (c >= inner_size || r >= inner_size || c < 0 || r < 0)
    {
      bc.row(pi) << -1, T(0.0), T(0.0);
      continue;
    } 
    const  VecC3<T> ul_3d = Vl.row(c + r*resolution);
    const  VecC3<T> br_3d = Vl.row(c + 1 + (r+1)*resolution);
    
    const VecC2<T> ul_reference(ul_3d(0), ul_3d(2));
    const VecC2<T> br_reference(br_3d(0), br_3d(2));
    
    const double ul_squared_dist = (ul_reference - current_point).squaredNorm();
    const double br_squared_dist = (br_reference - current_point).squaredNorm();
    
    VecC2<T> v_a;
    VecC2<T> v_b;
    VecC2<T> v_c;
    
    int f_idx = -1;
    
    // Find corner vertices of hit triangle
    if (ul_squared_dist <= br_squared_dist)
    {
      f_idx = 2 * c + r * 2 * inner_size;
      
      v_a << Vl.row(c + r*resolution)(0), Vl.row(c + r*resolution)(2);
      v_b << Vl.row(c+1 + r*resolution)(0),Vl.row(c+1 + r*resolution)(2);
      v_c << Vl.row(c + (r+1)*resolution)(0),Vl.row(c + (r+1)*resolution)(2);
    }else
    {
      f_idx = 2 * c + r * 2 * inner_size + 1;
      
      v_a << Vl.row(c+1 + (r+1)*resolution)(0),Vl.row(c+1 + (r+1)*resolution)(2);
      v_b << Vl.row(c + (r+1)*resolution)(0),Vl.row(c + (r+1)*resolution)(2);
      v_c << Vl.row(c+1 + r*resolution)(0),Vl.row(c+1 + r*resolution)(2);
    }
    
    T u,v,w;
    barycentric(current_point, v_a, v_b, v_c, u, v, w);
    
    bc.row(pi) << T(f_idx), u, v;
  }
}

template <typename T>
void Mesh<T>::set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  current_target_point_cloud = P;
  
  residuals.push_back(std::vector<std::vector<T>>(MESH_LEVELS-1));
  
  // First fuse to base level
  Mat<T> bc;
  project_points(0, bc);
  update_JtJ(0, bc); // Update lh
  update_Jtz(0, bc, current_target_point_cloud.col(1));
  
  for (int li = 1; li < MESH_LEVELS; ++li)
  { 
    Mat<T> V_upsampled;
    Eigen::MatrixXi F_upsampled;
    
    get_mesh(li, V_upsampled, F_upsampled);
    project_points(li, bc);
    
    update_JtJ(li, bc); // Update lh
  }  
}

template <typename T>
void Mesh<T>::set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C)
{ 
  set_target_point_cloud(P);
  
  if (C.rows() != P.rows())
    return;
    
  const VecC3<T> bb_min = V[0].row(0);
  const VecC3<T> bb_max = V[0].row(V[0].rows()-1);

  const T cdx = (bb_max(0)-bb_min(0)) / static_cast<T>(TEXTURE_RESOLUTION-1);
  const T cdz = (bb_max(2)-bb_min(2)) / static_cast<T>(TEXTURE_RESOLUTION-1);

  for(int i=0; i<current_target_point_cloud.rows(); i++)
  { 
    T x = current_target_point_cloud(i, 0) - bb_min(0);
    T z = current_target_point_cloud(i, 2) - bb_min(2);
    x = std::floor(x / cdx);
    z = std::floor(z / cdz);
    
    const int xi = static_cast<int>(x);
    const int zi = static_cast<int>(z);
    
    if (xi >= color_counter.rows() || zi >= color_counter.cols() || xi < 0 || zi < 0)
      continue;

    const int count = color_counter(xi, zi);
        
    texture.red(xi, zi)   = static_cast<unsigned char>((texture.red(xi, zi)  * count + (C(i,0)*255)) / (count + 1));
    texture.green(xi, zi) = static_cast<unsigned char>((texture.green(xi, zi) * count + (C(i,1)*255)) / (count + 1));
    texture.blue(xi, zi)  = static_cast<unsigned char>((texture.blue(xi, zi)  * count + (C(i,2)*255)) / (count + 1));
    ++color_counter(xi, zi);
  }
}

template <typename T>
void Mesh<T>::update_JtJ(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc)
{    
  for (int i = 0; i < bc.rows(); ++i)
  {
    const VecR3<T>& row = bc.row(i);

    if (static_cast<int>(row(0)) == -1)
      continue;
    
    JtJ[level].update_triangle(static_cast<int>(row(0)), row(1), row(2));
  }
}

template <typename T>
void Mesh<T>::update_Jtz(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc, const Eigen::Matrix<T, Eigen::Dynamic, 1>& z)
{    
  for (int i = 0; i < bc.rows(); ++i)
  {
    const VecR3<T>& row = bc.row(i);

    if (static_cast<int>(row(0)) == -1)
      continue;
    
    Jtz[level].update_triangle(static_cast<int>(row(0)), row(1), row(2), z(i));
  }
}
  
template <typename T>
void Mesh<T>::solve(const int iterations)
{
  
  // First fuse to ba se level
  Mat<T> bc;

// lh and rh for first layer already updated in set_target_point_cloud
#ifdef ENABLE_CUDA
  sor_gpu(iterations, 0, V[0].col(1));
#else
  sor_parallel(iterations, 0, V[0].col(1));
#endif
  
  for (int li = 1; li < MESH_LEVELS; ++li)
  {
    Mat<T> residual_height = Mat<T>::Zero(current_target_point_cloud.rows(), current_target_point_cloud.cols());
    
    Mat<T> V_upsampled;
    Eigen::MatrixXi F_upsampled;
    get_mesh(li, V_upsampled, F_upsampled);
    project_points(li, bc);
    
    // Compute residual
#pragma omp parallel for
    for (int pi = 0; pi < bc.rows(); ++pi)
    {
      if (static_cast<int>(bc.row(pi)(0)) == -1) // No hit
        continue;
      
      const VecC3<T> v0 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(0));
      const VecC3<T> v1 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(1));
      const VecC3<T> v2 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(2));
      
      // Ideally these should be equal, then there is no error
      const VecC3<T> solved_point = bc.row(pi)(1) * v0 + bc.row(pi)(2) * v1 + (1.f - bc.row(pi)(1) - bc.row(pi)(2)) * v2;
      const VecC3<T> measured_point = current_target_point_cloud.row(pi);
      
      residual_height.row(pi) = measured_point - solved_point;
    }
    
    residuals.back()[li-1].push_back(residual_height.array().sum());
    
    // Update equation rh with residual
    update_Jtz(li, bc, residual_height.col(1));

#ifdef ENABLE_CUDA
      sor_gpu(iterations, li, V[li].col(1));
#else
      sor_parallel(iterations, li, V[li].col(1));
#endif
  }
}

// Basic solve without any parallelism
template <typename T>
void Mesh<T>::sor(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz[level];
  const auto& Jtj_mat = JtJ[level];
  
  // For each iteration, solve system for each vertex
  for(int it = 0; it < iterations; it++)
  {
    for (int vi = 0; vi < h.rows(); vi++)
      sor_inner(vi, Jtj_mat, Jtz_vec, h.data());
  }
}

/* 
 Solve system for a single vertex vi in the mesh
 vi is the vertex index,
 JtJ is the MatrixGrid, matrix on the lhs
 Jtz is the vector on the rhs
 h is an array of the vertex heights
*/
template <typename T>
CUDA_HOST_DEVICE inline void sor_inner(const int vi, const JtJMatrixGrid<T>& JtJ, const JtzVector<T>& Jtz_vec, T* h)
{
  T xn = Jtz_vec.get(vi);
  T acc = 0;

  T vals[6];
  int ids[6];

  T a;
  JtJ.get_matrix_values_for_vertex(vi, vals, ids, a);

  for (int j = 0; j < 6; ++j)
    acc += vals[j] * h[ids[j]];

  xn -= acc;

  // Weighting of previous height vs newly solved height
  // only use new height if w = 1.
  const T w = 1.0;
  h[vi] = (1.f-w) * h[vi] + w*xn/a;
}

// Run solve in parallel by using four-color reordering
template <typename T>
void Mesh<T>::sor_parallel(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz[level];
  const auto& Jtj_mat = JtJ[level];
  const int resolution = JtJ[level].get_mesh_width();
    
  for(int it = 0; it < iterations; it++)
  {
#pragma omp parallel for collapse(2)
    for (int x = 0; x < resolution; x+=2)
      for (int y = 0; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, Jtj_mat, Jtz_vec, h.data());
      }

#pragma omp parallel for collapse(2)    
    for (int x = 1; x < resolution; x+=2)
      for (int y = 0; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, Jtj_mat, Jtz_vec, h.data());
      }

#pragma omp parallel for collapse(2)
    for (int x = 0; x < resolution; x+=2)
      for (int y = 1; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, Jtj_mat, Jtz_vec, h.data());
      }
      
#pragma omp parallel for collapse(2)
    for (int x = 1; x < resolution; x+=2)
      for (int y = 1; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, Jtj_mat, Jtz_vec, h.data());
      }
  }
}

// Define CUDA specific functions here
#ifdef ENABLE_CUDA

// This function solves a single iteration for a point based on the index
// xoffset and yoffset define the offset of the four-color reordering
__global__ void solve_kernel(const int xoffset, const int yoffset, const int mesh_width, const JtzVector<float> Jtz_vec, const JtJMatrixGrid<float> JtJ_mat, float* h) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (idx >= mesh_width*mesh_width)
    {
	    return;
    }

	const int x = ((idx) % mesh_width);
	const int y = ((idx) / mesh_width);
	const int vi = x + y * mesh_width;

  // Make sure that indices is okay for our offsets
	if ((x % 2 == xoffset) && (y % 2 == yoffset))
		sor_inner(vi, JtJ_mat, Jtz_vec, h);
}

// GPU solve for non-float meshes, does not work
template <typename T>
void Mesh<T>::sor_gpu(const int, const int, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>)
{
    assert(false && "GPU solver works only with float meshes");
}

/*
 Function for solving the system on the GPU using CUDA
 Similar to the parallel solve this method uses four-color reordering
 and therefore there are four solve_kernel calls,
 GPU solver also only works with float meshes
*/ 
template <>
void Mesh<float>::sor_gpu(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<float, Eigen::Dynamic, 1>> h)
{
    const JtJMatrixGrid<float> JtJ_mat = JtJ[level];
    const JtzVector<float> Jtz_vec = Jtz[level];
    const int mesh_width = JtJ[level].get_mesh_width();
    const int mesh_width_squared = mesh_width*mesh_width;
    const int mat_width = JtJ[level].get_matrix_width();
    //const int mat_width_squared = mat_width*mat_width;

    float *devH;
    CUDA_CHECK(cudaMalloc(&devH, h.rows() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(devH, h.data(), h.rows() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(64);
    dim3 grid((mesh_width_squared + block.x - 1) / block.x);

    for (int it = 0; it < iterations; it++) {
	    solve_kernel<<<grid, block>>>(0, 0, mesh_width, Jtz_vec, JtJ_mat, devH);
	    cudaDeviceSynchronize();
	    solve_kernel<<<grid, block>>>(1, 0, mesh_width, Jtz_vec, JtJ_mat, devH);
	    cudaDeviceSynchronize();
	    solve_kernel<<<grid, block>>>(0, 1, mesh_width, Jtz_vec, JtJ_mat, devH);
	    cudaDeviceSynchronize();
	    solve_kernel<<<grid, block>>>(1, 1, mesh_width, Jtz_vec, JtJ_mat, devH);
	    cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaMemcpy(h.data(), devH, h.rows() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devH));
}
#endif
