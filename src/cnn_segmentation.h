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

#ifndef CNN_SEGMENTATION_H_  // NOLINT
#define CNN_SEGMENTATION_H_  // NOLINT

#include <memory>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "config.h"
#include "pcl_types.h"
#include "object.h"
#include "cluster2d.h"
#include "feature_generator.h"

namespace apollo {
namespace perception {

class CNNSegmentation {
 public:
  CNNSegmentation() {}
  ~CNNSegmentation() {}

  bool Init();

  bool Segment(pcl_util::PointCloudPtr pc_ptr,
               const pcl_util::PointIndices& valid_indices,
               std::vector<std::shared_ptr<Object>>* objects);

  std::string name() const { return "CNNSegmentation"; }

  float range() const { return range_; }
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  // range of bird-view field (for each side)
  float range_ = 0.0;
  // number of cells in bird-view width
  int width_ = 0;
  // number of cells in bird-view height
  int height_ = 0;

  // paramters of CNNSegmentation
  // Caffe network object
  std::shared_ptr<caffe::Net<float>> caffe_net_;

  // bird-view raw feature generator
  std::shared_ptr<cnnseg::FeatureGenerator<float>> feature_generator_;

  // center offset prediction
  boost::shared_ptr<caffe::Blob<float>> instance_pt_blob_;
  // objectness prediction
  boost::shared_ptr<caffe::Blob<float>> category_pt_blob_;
  // fg probability prediction
  boost::shared_ptr<caffe::Blob<float>> confidence_pt_blob_;
  // object height prediction
  boost::shared_ptr<caffe::Blob<float>> height_pt_blob_;
  // raw features to be input into network
  boost::shared_ptr<caffe::Blob<float>> feature_blob_;
  // class prediction
  boost::shared_ptr<caffe::Blob<float>> class_pt_blob_;

  // use all points of cloud to compute features
  bool use_full_cloud_ = false;

  // clustering model for post-processing
  std::shared_ptr<cnnseg::Cluster2D> cluster2d_;
};


}  // namespace perception
}  // namespace apollo

#endif
