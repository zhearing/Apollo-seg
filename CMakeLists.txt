cmake_minimum_required(VERSION 3.9)
project(Apollo_seg)
set(CMAKE_CXX_STANDARD 11)

set(Caffe_DIR /home/zyzhong/git/caffe)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
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

add_executable(Apollo_seg
        src/config.h
        src/base_object_builder.h
        src/cluster2d.h
        src/cnn_segmentation.cc
        src/cnn_segmentation.h
        src/convex_hullxy.h
        src/disjoint_set.h
        src/feature_generator.cc
        src/feature_generator.h
        src/geometry_util.cc
        src/geometry_util.h
        src/macro.h
        src/min_box.cc
        src/min_box.h
        src/object.h
        src/pcl_types.h
        src/types.cc
        src/types.h
        src/util.h)
        
target_link_libraries(Apollo_seg
    ${Caffe_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Eigen_LIBRARIES}
)
