
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    set_source_files_properties(main.cpp Util.cpp PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --Werror cross-execution-space-call -Xptxas -dlcm=ca --ptxas-options=-v --compiler-options=\"${CXX_FLAGS}\"")
endif(ENABLE_CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-error=unused-parameter -Wconversion -Wno-error=conversion")


add_executable(${PROJECT_NAME} ${CXX_SRC})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${PCL_LIBRARIES} igl::core igl::opengl_glfw igl::opengl igl::opengl_glfw_imgui)

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
