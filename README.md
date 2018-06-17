# rtsr
rtsr as in Real Time Surface Reconstruction.  

This project is an effort to implement the method described in this paper:  
J. Zienkiewicz, A. Tsiotsios, A. Davison, and S. Leutenegger. Monocular, real-time surface reconstruction using dynamic level of detail. In 2016 Fourth International Conference on 3D Vision (3DV), pages 37–46, Oct 2016  

This project relies on a number of external libraries. Fortunately, most of them can be easily included via the geometry processing library IGL. IGL has a number of dependent submodules. Therefore, to clone this project correctly, issue  

```
git clone --recursive https://github.com/asterycs/rtsr.git
```

# Additional dependencies
The project has a CUDA extension, and this is handled in such a way that a fairly recent CMake version is required. This project has been tested with CMake v3.10.2 but others might work as well. Adjust the required version in CMakeLists.txt if needed.  

We use PCL for principal plane detection in the point clouds. Therefore you need to have the PCL development files prior to build for principal plane detection to work. If CMake doesn't detect pcl, these features are left out.  

On Ubuntu and similar:  
Do NOT install pcl via the package manager. PCL depends on an Eigen version that will be selected by CMake, but it is not compatible with this project. Build it from source and point CMake to the correct installation directory.  

Additionally, for the CUDA features you need the CUDA toolkit. The project has been compiled with versions 9.1 and 9.2. The CUDA toolkit can be obtained from []{https://developer.nvidia.com/cuda-downloads}.  


# Build

```
cd rtsr
mkdir build
cd build/
cmake ../
make
```

You can activate CUDA support by specifying -DENABLE_CUDA=ON when running CMake. Note that there is a huge performance impact for compiling in debug mode.  


# Layout

src contains all the relevant source code.  
DataSet.hpp/cpp contains a class for reading the kinect data  
EqHelpers.hpp/tpp contain helper classes for managing the linear system  
Mesh.hpp/tpp contains the mesh abstraction  
Util.hpp/cpp contains CUDA related error checking  
