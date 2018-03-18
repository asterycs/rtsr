
file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    file(GLOB CU_SRC *.cu)
    #set_source_files_properties(CUDA_CPP_SOURCE_FILES_HERE PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-error=unused-parameter")

add_executable(${PROJECT_NAME} ${CXX_SRC} ${CU_SRC})
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::triangle igl::embree)

if (ENABLE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(ENABLE_CUDA)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS filesystem)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()
