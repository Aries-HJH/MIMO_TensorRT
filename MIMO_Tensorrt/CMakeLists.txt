cmake_minimum_required(VERSION 2.6)

project(MIMO_tensorrt)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

include_directories(/usr/include/aarch64-linux-gun/)
link_directories(/usr/lib/aarch64-linux-gnu/)

# PROJECT_SOUCE_DIR is the folder where the most recent CMakeLists.txt file containing PROJECT() is located.
add_executable(MIMO_tensorrt ${PROJECT_SOURCE_DIR}/MIMO_Tensorrt.cpp)
target_link_libraries(MIMO_tensorrt nvinfer)
target_link_libraries(MIMO_tensorrt nvonnxparser)
target_link_libraries(MIMO_tensorrt cudart)

