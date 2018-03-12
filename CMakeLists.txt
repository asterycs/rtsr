cmake_minimum_required(VERSION 3.10)
project(rtsr)

# Parameters
set(ENABLE_CUDA OFF CACHE BOOL "Enable Cuda features") # off by default for now

# Set project language
if (ENABLE_CUDA)
    project(rtsr LANGUAGES CXX CUDA)
    add_definitions(-DENABLE_CUDA)
else()
    project(rtsr LANGUAGES CXX)
    message("CUDA features disabled")
endif(ENABLE_CUDA)

include(ExternalProject)
ExternalProject_Add(libigl
    PREFIX "${CMAKE_BINARY_DIR}/ext/libigl"
    GIT_PROGRESS 1
    GIT_REPOSITORY https://github.com/libigl/libigl.git 
    UPDATE_DISCONNECTED 1
    LOG_DOWNLOAD 1
    LOG_BUILD 1
    LOG_INSTALL 1
    CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR>/optional
    INSTALL_COMMAND "" # Libigl doesn't provide install command :(
    CMAKE_ARGS
       "LIBIGL_USE_STATIC_LIBRARY=ON"
       "-DCMAKE_BUILD_TYPE=Release"
       )

ExternalProject_Get_Property(libigl SOURCE_DIR)

set(LIBIGL_EXTRA_INCLUDE_DIRS
    ${SOURCE_DIR}/include
    ${SOURCE_DIR}/external/eigen/
    # add more of libigl's dependencies if needed
    )

ExternalProject_Get_Property(libigl BINARY_DIR)

set(LIBIGL_LIBRARIES
    ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}glad${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}glfw3${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}igl_core${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}igl_opengl_glfw${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}triangle${CMAKE_STATIC_LIBRARY_SUFFIX}
    )


add_subdirectory("src/")
