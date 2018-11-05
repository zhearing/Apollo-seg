/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "cnn_segmentation.h"

#define USE_GPU

namespace apollo {
namespace perception {

bool CNNSegmentation::Init() {
  std::string proto_file = PROTO_FILE;
  std::string weight_file = WEIGHT_FILE;
  
  range_ = 60.0;
  width_ = 640;
  height_ = 640;

/// Instantiate Caffe net
#ifndef USE_GPU
  std::cout << "using Caffe CPU mode";
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  std::cout << "using Caffe GPU mode";
  int gpu_id = 0;
  CHECK_GE(gpu_id, 0);
  caffe::Caffe::SetDevice(gpu_id);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::DeviceQuery();
#endif

  caffe_net_.reset(new caffe::Net<float>(proto_file, caffe::TEST));
  caffe_net_->CopyTrainedLayersFrom(weight_file);

  /// set related Caffe blobs
  // center offset prediction
  std::string instance_pt_blob_name = "instance_pt";
  instance_pt_blob_ = caffe_net_->blob_by_name(instance_pt_blob_name);
  CHECK(instance_pt_blob_ != nullptr) << "`" << instance_pt_blob_name
                                      << "` not exists!";
  // objectness prediction
  std::string category_pt_blob_name = "category_score";
  category_pt_blob_ = caffe_net_->blob_by_name(category_pt_blob_name);
  CHECK(category_pt_blob_ != nullptr) << "`" << category_pt_blob_name
                                      << "` not exists!";
  // positiveness (foreground object probability) prediction
  std::string confidence_pt_blob_name = "confidence_score";
  confidence_pt_blob_ = caffe_net_->blob_by_name(confidence_pt_blob_name);
  CHECK(confidence_pt_blob_ != nullptr) << "`" << confidence_pt_blob_name
                                        << "` not exists!";
  // object height prediction
  std::string height_pt_blob_name = "height_pt";
  height_pt_blob_ = caffe_net_->blob_by_name(height_pt_blob_name);
  CHECK(height_pt_blob_ != nullptr) << "`" << height_pt_blob_name
                                    << "` not exists!";
  // raw feature data
  std::string feature_blob_name = "data";
  feature_blob_ = caffe_net_->blob_by_name(feature_blob_name);
  CHECK(feature_blob_ != nullptr) << "`" << feature_blob_name
                                  << "` not exists!";
  // class prediction
  std::string class_pt_blob_name = "class_score";
  class_pt_blob_ = caffe_net_->blob_by_name(class_pt_blob_name);
  CHECK(class_pt_blob_ != nullptr) << "`" << class_pt_blob_name
                                   << "` not exists!";

  cluster2d_.reset(new cnnseg::Cluster2D());
  if (!cluster2d_->Init(height_, width_, range_)) {
    std::cerr << "Fail to Init cluster2d for CNNSegmentation";
  }

  feature_generator_.reset(new cnnseg::FeatureGenerator<float>());
  if (!feature_generator_->Init(feature_blob_.get())) {
    std::cerr << "Fail to Init feature generator for CNNSegmentation";
    return false;
  }

  return true;
}

bool CNNSegmentation::Segment(pcl_util::PointCloudPtr pc_ptr,
                              const pcl_util::PointIndices& valid_indices,
                              std::vector<std::shared_ptr<Object>>* objects) {
  objects->clear();
  int num_pts = static_cast<int>(pc_ptr->points.size());
  if (num_pts == 0) {
    std::cout << "None of input points, return directly.";
    return true;
  }

  use_full_cloud_ = false;

  // generate raw features
  feature_generator_->Generate(pc_ptr);

// network forward process
#ifdef USE_GPU
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  caffe_net_->Forward();
  std::cout << "[CNNSeg] CNN forward" << std::endl;

  // clutser points and construct segments/objects
  float objectness_thresh = 0.5;
  bool use_all_grids_for_clustering = false;
  cluster2d_->Cluster(*category_pt_blob_, *instance_pt_blob_, pc_ptr,
                      valid_indices, objectness_thresh,
                      use_all_grids_for_clustering);
  std::cout << "[CNNSeg] clustering" << std::endl;

  caffe::Blob<float>* input_data_blob = feature_blob_.get();
  const float* input_data = input_data_blob->cpu_data();
  const float* input_count_data = input_data + input_data_blob->offset(0, 2);

  float filter_thresh = 30;
  float enable_filter_thresh = 700;
  cluster2d_->Filter(*confidence_pt_blob_, *height_pt_blob_, input_count_data,
                     filter_thresh, enable_filter_thresh);

  cluster2d_->Classify(*class_pt_blob_);

  float confidence_thresh = 0.1;
  float height_thresh = 0.5;
  int min_pts_num = 3;
  cluster2d_->GetObjects(confidence_thresh, height_thresh, min_pts_num, objects,
                         input_count_data);
  std::cout << "[CNNSeg] post-processing" << std::endl;

  return true;
}

}  // namespace perception
}  // namespace apollo
