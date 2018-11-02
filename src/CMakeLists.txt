cmake_minimum_required(VERSION 3.9)
project(Apollo_seg)
set(CMAKE_CXX_STANDARD 11)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV REQUIRED)

#Caffe
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

# Eigen
find_package(Eigen3 REQUIRED)


add_executable(Apollo_seg
        base_object_builder.h
        cluster2d.h
        cnn_segmentation.cc
        cnn_segmentation.h
        convex_hullxy.h
        feature_generator.cc
        feature_generator.h
        geometry_util.cc
        geometry_util.h
        macro.h
        min_box.cc
        min_box.h
        object.h
        pcl_types.h
        types.cc
        types.h
        util.h)
