
file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    file(GLOB CU_SRC *.cu)
    #set_source_files_properties(CUDA_CPP_SOURCE_FILES_HERE PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-error=unused-parameter -Wconversion -Wno-error=conversion")

add_executable(${PROJECT_NAME} ${CXX_SRC} ${CU_SRC})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} igl::core igl::opengl_glfw igl::triangle igl::embree)

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_result)
    if(ipo_result)
        set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif(CMAKE_BUILD_TYPE EQUAL "RELEASE")

if (ENABLE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(ENABLE_CUDA)
