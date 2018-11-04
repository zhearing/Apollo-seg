cmake_minimum_required(VERSION 3.9)
project(Apollo_seg)
set(CMAKE_CXX_STANDARD 11)

set(Caffe_DIR /home/zyzhong/git/caffe)
set(Caffe_INCLUDE_DIRS /home/zyzhong/git/caffe/include
        /home/zyzhong/git/caffe/build/lib)
set(PCL_INCLUDE_DIRS /usr/local/include/pcl-1.8)
set(CUDA_INCLUDE_DIRS /usr/local/cuda-9.0/include)

# PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4") # use this for ubuntu 16.04
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV REQUIRED)

#Caffe
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${Caffe_DEFINITIONS})

# Eigen
find_package(Eigen3 REQUIRED)

# CUDA
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR} /usr/local/lib /usr/lib)
link_directories(${PROJECT_BINARY_DIR}/libs)
#set lib output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
#set excutable output path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

#add subdir lib
add_library(cnn_seg src/feature_generator.cc
        src/cnn_segmentation.cc
        src/min_box.cc
        src/geometry_util.cc
        #src/fortracking.cc
        )
link_libraries(cnn_seg)

add_executable(Apollo_seg src/main.cc)

target_link_libraries(Apollo_seg
    ${Caffe_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Eigen_LIBRARIES}
    cnn_seg)
