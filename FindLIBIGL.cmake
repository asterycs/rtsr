find_path(LIBIGL_INCLUDE_DIR igl/readOBJ.h
    PATHS
        ${CMAKE_SOURCE_DIR}/libigl
    PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBIGL
    "\nlibigl not found --- run \"git submodule update --init --recursive\""
    LIBIGL_INCLUDE_DIR)
mark_as_advanced(LIBIGL_INCLUDE_DIR)

list(APPEND CMAKE_MODULE_PATH "${LIBIGL_INCLUDE_DIR}/../shared/cmake")
include(libigl)
