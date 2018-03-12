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
cd libigl/
git apply ../libigl_static_build.patch
cd ..
mkdir build
cd build/
cmake ../
make
```
