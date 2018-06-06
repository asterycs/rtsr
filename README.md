# rtsr
rtsr as in Real Time Surface Reconstruction.  

  
This project relies on a number of external libraries. Fortunately, most of them can be easily included via the geometry processing library IGL. IGL has a number of dependent submodules. Therefore, to clone this project correctly, issue

```
git clone --recursive https://github.com/asterycs/rtsr.git
```

# Additional dependencies
The project has a CUDA extension, and this is handled in such a way that a fairly recent CMake version is required. This project has been tested with CMake v3.10.2 but others might work as well. Adjust the required version in CMakeLists.txt if required.  

We use PCL for principal plane detection in the point clouds. Therefore you need to have the PCL development files prior to build.

On Ubuntu and similar:
```
apt install libpcl-dev
```
or build from source and point CMake to the correct installation directory.

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


17.5
Merged CUDA solver.
cmake flag -DENABLE_CUDA=ON enables the CUDA features.

3.5
Multiresolution fusion implemented according to the residual scheme explained in the paper. Use '0' and '+' to change rendered resolution.

24.4
Parallel successive over-relaxed solver implemented. Not sure about correctness but converges to similar solution as sequential version.

13.4
Demo with two point clouds for midterm presentation.

13.4
JtJ matrix is now represented as a dense grid, just like in the paper. See class "JtJMatrixGrid" in EqHelpers.hpp

28.3.2018
Gauss-Seidel works. Current point cloud is syntetically generated for clarity.

TODO:
Dataset parsing and transforming to world space.

13.3.2018:  
Started experimenting with this data -> https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_desk

./rtsr <path_to_data>
