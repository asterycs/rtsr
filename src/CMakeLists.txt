
file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    file(GLOB CU_SRC *.cu)
    #set_source_files_properties(CUDA_CPP_SOURCE_FILES_HERE PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

add_executable(${PROJECT_NAME} ${CXX_SRC} ${CU_SRC} ${LIBIGL_EXTRA_SOURCES})
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw stdc++fs)

if (ENABLE_CUDA)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(ENABLE_CUDA)
