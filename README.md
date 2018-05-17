# rtsr
rtsr as in Real Time Surface Reconstruction.

```
git clone --recurse-submodules -j4 https://github.com/asterycs/rtsr.git
```  
or
```
git clone --recursive https://github.com/asterycs/rtsr.git
```
```
cd rtsr
mkdir build
cd build/
cmake ../
make
```
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
