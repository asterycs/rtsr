#include "Mesh.hpp"

#include "igl/fit_plane.h"
#include "igl/mat_min.h"

#include <Eigen/QR>

#include <iostream>
#include <cstdlib>

template <typename T>
Mesh<T>::Mesh()
{
  color_counter =  Eigen::MatrixXi::Zero(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION);

  UV.resize(TEXTURE_RESOLUTION*TEXTURE_RESOLUTION, 2);
  texture_red.resize(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION); 
  texture_green.resize(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION);
  texture_blue.resize(TEXTURE_RESOLUTION, TEXTURE_RESOLUTION);
}

template <typename T>
Mesh<T>::~Mesh()
{
  
}

// There must be a formula for this
int subdivided_side_length(const int level, const int base_resolution)
{
  int res = base_resolution;
  
  for (int i = 0; i < level; ++i)
    res += res-1;
  
  return res;
}

template <typename T>
void upsample(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V, const Eigen::MatrixXi& F, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out)
{
    using TvecC3 = Eigen::Matrix<T, 3, 1>;
    const int old_resolution = static_cast<int>(std::sqrt(static_cast<double>(V.rows())));
    const int resolution = subdivided_side_length(1, old_resolution);
    const int faces = (resolution - 1) * (resolution - 1) * 2;
    const int vertices = resolution * resolution;
    
    F_out = Eigen::MatrixXi::Zero(faces, 3);
    V_out = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(vertices, 3);
        
    for (int y_step = 0; y_step < old_resolution-1; ++y_step)
    {
      for (int x_step = 0; x_step < old_resolution-1; ++x_step)
      {
        
        Eigen::Vector3i f_u = F.row(x_step*2 + y_step*(old_resolution-1)*2);
        Eigen::Vector3i f_l = F.row(x_step*2 + y_step*(old_resolution-1)*2 + 1);

        const TvecC3 old_v0 = V.row(f_u(0));
        const TvecC3 old_v1 = V.row(f_u(1));
        const TvecC3 old_v2 = V.row(f_u(2));
        
        const TvecC3 old_v3 = V.row(f_l(0));
        const TvecC3 old_v4 = V.row(f_l(1));
        const TvecC3 old_v5 = V.row(f_l(2));

        const TvecC3 v0 = old_v0;
        const TvecC3 v1 = (old_v0 + old_v1) * T(0.5);
        const TvecC3 v2 = old_v1;
        const TvecC3 v3 = (old_v0 + old_v2) * T(0.5);
        const TvecC3 v4 = (old_v2 + old_v1) * T(0.5);
        const TvecC3 v5 = (old_v2 + old_v5) * T(0.5);
        const TvecC3 v6 = old_v2;
        const TvecC3 v7 = (old_v3 + old_v4) * T(0.5);
        const TvecC3 v8 = old_v3;

        // TODO Check boundaries
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
        V_out.row(v2i) = v2;
        V_out.row(v3i) = v3;
        V_out.row(v4i) = v4;
        V_out.row(v5i) = v5;
        V_out.row(v6i) = v6;
        V_out.row(v7i) = v7;
        V_out.row(v8i) = v8;

        const int new_f_xi = x_step * 2;
        const int new_f_yi = y_step * 2;
        
        F_out.row(new_f_xi       + new_f_yi * (resolution-1)) << v0i,v1i,v3i;
        F_out.row(new_f_xi + 1 + new_f_yi * (resolution-1)) << v4i,v3i,v1i;
        F_out.row(new_f_xi + 2 + new_f_yi * (resolution-1)) << v1i,v2i,v4i;
        F_out.row(new_f_xi + 3 + new_f_yi * (resolution-1)) << v5i,v4i,v2i;
        F_out.row(new_f_xi +       (new_f_yi+1) * (resolution-1)) << v3i,v4i,v6i;
        F_out.row(new_f_xi + 1 + (new_f_yi+1) * (resolution-1)) << v7i,v6i,v4i;
        F_out.row(new_f_xi + 2 + (new_f_yi+1) * (resolution-1)) << v4i,v5i,v7i;
        F_out.row(new_f_xi + 3 + (new_f_yi+1) * (resolution-1)) << v8i,v7i,v5i;
        
        
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
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V_in = V;
    
    for (unsigned int i = 0; i < levels; ++i)
    {
      upsample(V_in, F_in, V_out, F_out);
      F_in = F_out;
      V_in = V_out;
    }
}

template <typename T>
template <int Rows, int Cols>
void Mesh<T>::align_to_point_cloud(const Eigen::Matrix<T, Rows, Cols>& P)
{  
  using TvecR3 = Eigen::Matrix<T, 1, 3>;
  using TvecC3 = Eigen::Matrix<T, 3, 1>;
  
  const TvecR3 bb_min = P.colwise().minCoeff();
  const TvecR3 bb_max = P.colwise().maxCoeff();
  const TvecR3 bb_d = (bb_max - bb_min).cwiseAbs();
  
  JtJ.resize(MESH_LEVELS);
  Jtz.resize(MESH_LEVELS);
  V.resize(MESH_LEVELS);
  F.resize(MESH_LEVELS);

  for (int li = 0; li < MESH_LEVELS; ++li)
  {  
    const double scaling_factor = MESH_SCALING_FACTOR;
    const int resolution = subdivided_side_length(li, MESH_RESOLUTION);
    
    // Scaling matrix
    const Eigen::Transform<T, 3, Eigen::Affine> scaling(Eigen::Scaling(TvecC3(scaling_factor*bb_d(0)/(resolution-1), 0.,scaling_factor*bb_d(2)/(resolution-1))));
    
    const TvecR3 pc_mean = P.colwise().mean();
    // P_centr: mean of the point cloud
    TvecR3 P_centr = bb_min + 0.5*(bb_max - bb_min);
    P_centr(1) = pc_mean(1); // Move to mean height w/r to pc instead of bb.
    
    const Eigen::Transform<T, 3, Eigen::Affine> t(Eigen::Translation<T, 3>(P_centr.transpose()));
    
    transform = t.matrix();
      
    V[li].resize(resolution*resolution, 3);
    F[li].resize((resolution-1)*(resolution-1)*2, 3);
    JtJ[li].resize(resolution);
    Jtz[li].resize(resolution);

  //#pragma omp parallel for
    for (int z_step = 0; z_step < resolution; ++z_step)
    {
      for (int x_step = 0; x_step < resolution; ++x_step)
      {
        Eigen::Matrix<T, 1, 4> v; v << TvecR3(x_step-T(resolution-1)/2.f,T(0.0), z_step-T(resolution-1)/2.f),T(1.0);
        Eigen::Matrix<T, 1, 3> pos;
        
        if (li == 0) // Only transform the first layer. Subsequent layers only denot a difference between the first
          pos << (v * scaling.matrix().transpose() * transform.transpose()).template head<3>();
        else
          pos << (v * scaling.matrix().transpose()).template head<3>();
   
        V[li].row(x_step + z_step*resolution) << pos;
      }
    }
    
  //#pragma omp parallel for
    for (int y_step = 0; y_step < resolution-1; ++y_step)
    {
      for (int x_step = 0; x_step < resolution-1; ++x_step)
      {
        F[li].row(x_step*2 + y_step*(resolution-1)*2)     << x_step+   y_step   *resolution,x_step+1+y_step*   resolution,x_step+(y_step+1)*resolution;
        F[li].row(x_step*2 + y_step*(resolution-1)*2 + 1) << x_step+1+(y_step+1)*resolution,x_step+ (y_step+1)*resolution,x_step+1+y_step*resolution;
      }
    }
     
   if (li == 0)
   {
      // Initialize Lh and rh with sensible values
      for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
        JtJ[li].update_triangle(i, 0.34f, 0.33f);

      for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
        Jtz[li].update_triangle(i, 0.34f, 0.33f, pc_mean(1));
   }else
   {
      for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
        JtJ[li].update_triangle(i, 0.34f, 0.33f);

      for (int i = 0; i < (resolution-1)*(resolution-1)*2; ++i)
        Jtz[li].update_triangle(i, 0.34f, 0.33f, T(0.0));
   }
  }
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

template <typename T>
void barycentric(const Eigen::Matrix<T, 2, 1>& p, const Eigen::Matrix<T, 2, 1>& a, const Eigen::Matrix<T, 2, 1>& b, const Eigen::Matrix<T, 2, 1>& c, T &u, T &v, T &w)
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

template <typename T>
void Mesh<T>::project_points(const int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc) const
{
  bc = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(current_target_point_cloud.rows(), 3);
  
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Vl;
  Eigen::MatrixXi Fl;
  get_mesh(level, Vl, Fl);
  
  const int resolution = subdivided_side_length(level, MESH_RESOLUTION);
  
  // Upper left triangle. Used to compute the index of the triangle that is hit
  const Eigen::Matrix<T, 2, 1> V_ul(Vl.row(0)(0), Vl.row(0)(2));
  const Eigen::Matrix<T, 2, 1> V_00(Vl.row(Fl.row(0)(0))(0), Vl.row(Fl.row(0)(0))(2));
  const Eigen::Matrix<T, 2, 1> V_01(Vl.row(Fl.row(0)(1))(0), Vl.row(Fl.row(0)(1))(2));
  const Eigen::Matrix<T, 2, 1> V_10(Vl.row(Fl.row(0)(2))(0), Vl.row(Fl.row(0)(2))(2));
  
  const double dx = (V_01- V_00).norm();
  const double dy = (V_10 - V_00).norm();
  
//#pragma omp parallel for
  for (int pi = 0; pi < current_target_point_cloud.rows(); ++pi)
  {
    const Eigen::Matrix<T, 2, 1> current_point(current_target_point_cloud.row(pi)(0), current_target_point_cloud.row(pi)(2));
    const Eigen::Matrix<T, 2, 1> offset(current_point - V_ul);
    
    const int c = static_cast<int>(offset(0) / dx);
    const int r = static_cast<int>(offset(1) / dy);
    
    const int inner_size = resolution - 1;
    if (c >= inner_size || r >= inner_size)
    {
      bc.row(pi) << -1, T(0.0), T(0.0);
      continue;
    } 
    const  Eigen::Matrix<T, 3, 1> ul_3d = Vl.row(c + r*resolution);
    const  Eigen::Matrix<T, 3, 1> br_3d = Vl.row(c + 1 + (r+1)*resolution);
    
    const Eigen::Matrix<T, 2, 1> ul_reference(ul_3d(0), ul_3d(2));
    const Eigen::Matrix<T, 2, 1> br_reference(br_3d(0), br_3d(2));
    
    const double ul_squared_dist = (ul_reference - current_point).squaredNorm();
    const double br_squared_dist = (br_reference - current_point).squaredNorm();
    
    Eigen::Matrix<T, 2, 1> v_a;
    Eigen::Matrix<T, 2, 1> v_b;
    Eigen::Matrix<T, 2, 1> v_c;
    
    int f_idx = -1;
    
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
    
    bc.row(pi) << f_idx, u, v;
  }
  
}

template <typename T>
void Mesh<T>::set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  current_target_point_cloud = P;
}
template <typename T>
void Mesh<T>::set_target_point_cloud(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& P, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& C)
{ 
  current_target_point_cloud = P;
  current_target_point_cloud_color = C;
}

template <typename T>
void Mesh<T>::update_weights(const int level, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& bc, const Eigen::Matrix<T, Eigen::Dynamic, 1>& z)
{    
  for (int i = 0; i < bc.rows(); ++i)
  {
    const Eigen::Matrix<T, 1, 3>& row = bc.row(i);

    if (static_cast<int>(row(0)) == -1)
      continue;
    
    JtJ[level].update_triangle(static_cast<int>(row(0)), row(1), row(2));
    Jtz[level].update_triangle(static_cast<int>(row(0)), row(1), row(2), z(i));
  }
}

template <typename T>
void Mesh<T>::debug(igl::opengl::glfw::Viewer &viewer)
{
      using TvecC3 = Eigen::Matrix<T, 3, 1>;
    using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;    
    Eigen::MatrixXd C(1, 3); C << 1.0, 0., 0.;
    viewer.data().clear();
    viewer.data().point_size = 5;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    get_mesh(1, vertices, faces);
    viewer.data().add_points(current_target_point_cloud.row(0), C);


        TMat bc;
    project_points(1, bc);

    TvecC3 v0 = vertices.row(faces.row(static_cast<int>(bc.row(0)(0)))(0));
    TvecC3 v1 = vertices.row(faces.row(static_cast<int>(bc.row(0)(0)))(1));
    TvecC3 v2 = vertices.row(faces.row(static_cast<int>(bc.row(0)(0)))(2));
    
    Eigen::MatrixXd points(3,3);
    points.row(0) << v0.transpose();
    points.row(1) << v1.transpose();
    points.row(2) << v2.transpose();
    viewer.data().add_points(points, C);
    
    viewer.data().set_mesh(vertices, faces);
    viewer.data().compute_normals();
    viewer.data().invert_normals = true;
    
   /* Eigen::MatrixXd C(1, 3); C << 1.0, 0., 0.;
    viewer.data().clear();
    viewer.data().point_size = 5;
    
        Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces;
    
    get_mesh(0, vertices, faces);
   TvecC3 v0 = vertices.row(faces.row(static_cast<int>(0))(0));
   TvecC3 v1 = vertices.row(faces.row(static_cast<int>(0))(1));
   TvecC3  v2 = vertices.row(faces.row(static_cast<int>(0))(2));
    
    TMat points(3,3);
    points.row(0) << v0.transpose();
    points.row(1) << v1.transpose();
    points.row(2) << v2.transpose();
    std::cout << points << std::endl;
    
    viewer.data().add_points(points, C);
    
        get_mesh(1, vertices, faces);
    v0 = vertices.row(faces.row(static_cast<int>(0))(0));
    v1 = vertices.row(faces.row(static_cast<int>(0))(1));
    v2 = vertices.row(faces.row(static_cast<int>(0))(2));
    
    points.resize(3,3);
    points.row(0) << v0.transpose();
    points.row(1) << v1.transpose();
    points.row(2) << v2.transpose();
    std::cout << points << std::endl;

    viewer.data().add_points(points, C);*/
    
}

template <typename T>
void Mesh<T>::solve(const int iterations)
{
  using TMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using TvecC3 = Eigen::Matrix<T, 3, 1>;
  
  // First fuse to base level
  TMat bc;
  project_points(0, bc);
  update_weights(0, bc, current_target_point_cloud.col(1));
  sor_parallel(iterations, 0, V[0].col(1));
  
  for (int li = 1; li < MESH_LEVELS; ++li)
  {
    TMat point_with_residual_height = TMat::Zero(current_target_point_cloud.rows(), current_target_point_cloud.cols());
    
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> V_upsampled;
    Eigen::MatrixXi F_upsampled;
    get_mesh(li, V_upsampled, F_upsampled);
    project_points(li, bc);
    
    for (int pi = 0; pi < bc.rows(); ++pi)
    {
      if (static_cast<int>(bc.row(pi)(0)) == -1) // No hit
        continue;
      
      const TvecC3 v0 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(0));
      const TvecC3 v1 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(1));
      const TvecC3 v2 = V_upsampled.row(F_upsampled.row(static_cast<int>(bc.row(pi)(0)))(2));
      
      // Ideally these should be equal
      const TvecC3 solved_point = bc.row(pi)(1) * v0 + bc.row(pi)(2) * v1 + (1.0 - bc.row(pi)(1) - bc.row(pi)(2)) * v2;
      const TvecC3 measured_point = current_target_point_cloud.row(pi);
      
      if (pi == 0)
      {
        //std::cout << measured_point << std::endl << solved_point<< std::endl << std::endl;
        //std::cout << v0 << std::endl << v1 <<std::endl << v2 <<std::endl <<std::endl;
      }
      
      point_with_residual_height.row(pi) = measured_point - solved_point;
    }
    
    update_weights(li, bc, point_with_residual_height.col(1));
    sor_parallel(iterations, li, V[li].col(1));
  }
}

template <typename T>
void Mesh<T>::sor(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz[level].get_vec();
    
  for(int it = 0; it < iterations; it++)
  {
    for (int vi = 0; vi < h.rows(); vi++)
      sor_inner(vi, JtJ[level], Jtz_vec, h);
  }
}

template <typename T>
inline void sor_inner(const int vi, const JtJMatrixGrid<T>& JtJ, const Eigen::Matrix<T, Eigen::Dynamic, 1>& Jtz_vec, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h)
{
  T xn = Jtz_vec(vi);
  T acc = 0;

  std::array<T, 6> vals;
  std::array<int, 6> ids;

  T a;
  JtJ.get_matrix_values_for_vertex(vi, vals, ids, a);

  for (int j = 0; j < 6; ++j)
    acc += vals[j] * h(ids[j]);

  xn -= acc;

  const T w = 1.0;
  h(vi) = (1-w) * h(vi) + w*xn/a;
}

template <typename T>
void Mesh<T>::sor_parallel(const int iterations, const int level, Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> h) const
{
  const auto& Jtz_vec = Jtz[level].get_vec();
  const int resolution = subdivided_side_length(level, MESH_RESOLUTION);
    
  for(int it = 0; it < iterations; it++)
  {
//#pragma omp parallel for collapse(2)
    for (int x = 0; x < resolution; x+=2)
      for (int y = 0; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, JtJ[level], Jtz_vec, h);
      }

//#pragma omp parallel for collapse(2)    
    for (int x = 1; x < resolution; x+=2)
      for (int y = 0; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, JtJ[level], Jtz_vec, h);
      }

//#pragma omp parallel for collapse(2)
    for (int x = 0; x < resolution; x+=2)
      for (int y = 1; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, JtJ[level], Jtz_vec, h);
      }
      
//#pragma omp parallel for collapse(2)
    for (int x = 1; x < resolution; x+=2)
      for (int y = 1; y < resolution; y+=2)
      {
        const int vi = x + y * resolution;
        sor_inner(vi, JtJ[level], Jtz_vec, h);
      }
  }
}

template <typename T>
void Mesh<T>::get_mesh(const unsigned int level, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& V_out, Eigen::MatrixXi& F_out, ColorData& colordata) 
{
    const Eigen::VectorXd bb_min = V[0].colwise().minCoeff();
    const Eigen::VectorXd bb_max = V[0].colwise().maxCoeff();

    const T cdx = (bb_max(0)-bb_min(0)) / static_cast<T>(TEXTURE_RESOLUTION-1);
    const T cdz = (bb_max(2)-bb_min(2)) / static_cast<T>(TEXTURE_RESOLUTION-1);

    for(int i=0; i<current_target_point_cloud.rows(); i++)
    { 
      T x = current_target_point_cloud(i, 0) - bb_min(0);
      T z = current_target_point_cloud(i, 2) - bb_min(2);
      x = std::floor(x / cdx);
      z = std::floor(z / cdz);
      
      if (x >= color_counter.rows() || z >= color_counter.cols() || x < 0 || z < 0)
        continue;
        
      int count = color_counter(x, z);

      texture_red(x, z) = (texture_red(x, z) * count + (current_target_point_cloud_color(i, 0)*255)) / (count + 1);
      texture_green(x, z) = (texture_green(x, z) * count + (current_target_point_cloud_color(i, 1)*255)) / (count + 1);
      texture_blue(x, z) = (texture_blue(x, z) * count + (current_target_point_cloud_color(i, 2)*255)) / (count + 1);
      color_counter(x, z) = color_counter(x, z) + 1;
    }

    //set UV coordinates
    const T tdx = 1.0 / (TEXTURE_RESOLUTION-1);
    for(int i = 0; i < TEXTURE_RESOLUTION; i++)
    {
      for(int j = 0; j < TEXTURE_RESOLUTION; j++)
      {
        int index = (j) * TEXTURE_RESOLUTION + i;
        UV(index, 0) = i*tdx;
        UV(index, 1) = j*tdx;
      }
    }
    colordata.UV = UV;
    colordata.texture_red = texture_red;
    colordata.texture_green = texture_green;
    colordata.texture_blue = texture_blue;

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
