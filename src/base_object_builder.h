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
#ifndef SRC_BASE_OBJECT_BUILDER_H_
#define SRC_BASE_OBJECT_BUILDER_H_

#include <string>
#include <vector>

#include "macro.h"
#include "geometry_util.h"
#include "pcl_types.h"
#include "object.h"

namespace apollo {
namespace perception {

struct ObjectBuilderOptions {
  Eigen::Vector3d ref_center;
};

class BaseObjectBuilder {
 public:
  BaseObjectBuilder() {}
  virtual ~BaseObjectBuilder() {}

  virtual bool Init() = 0;

  // @brief: calc object feature, and fill fields.
  // @param [in]: options.
  // @param [in/out]: object list.
  virtual bool Build(const ObjectBuilderOptions& options,
                     std::vector<std::shared_ptr<Object>>* objects) = 0;

  virtual std::string name() const = 0;

 protected:
  virtual void SetDefaultValue(pcl_util::PointCloudPtr cloud,
                               std::shared_ptr<Object> obj,
                               Eigen::Vector4f* min_pt,
                               Eigen::Vector4f* max_pt) {
    GetCloudMinMax3D<pcl_util::Point>(cloud, min_pt, max_pt);
    Eigen::Vector3f center(((*min_pt)[0] + (*max_pt)[0]) / 2,
                           ((*min_pt)[1] + (*max_pt)[1]) / 2,
                           ((*min_pt)[2] + (*max_pt)[2]) / 2);

    // handle degeneration case
    float epslin = 1e-3;
    for (int i = 0; i < 3; i++) {
      if ((*max_pt)[i] - (*min_pt)[i] < epslin) {
        (*max_pt)[i] = center[i] + epslin / 2;
        (*min_pt)[i] = center[i] - epslin / 2;
      }
    }

    // length
    obj->length = (*max_pt)[0] - (*min_pt)[0];
    // width
    obj->width = (*max_pt)[1] - (*min_pt)[1];
    if (obj->length - obj->width < 0) {
      float tmp = obj->length;
      obj->length = obj->width;
      obj->width = tmp;
      obj->direction = Eigen::Vector3d(0.0, 1.0, 0.0);
    } else {
      obj->direction = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    // height
    obj->height = (*max_pt)[2] - (*min_pt)[2];
    // center
    obj->center = Eigen::Vector3d(((*max_pt)[0] + (*min_pt)[0]) / 2,
                                  ((*max_pt)[1] + (*min_pt)[1]) / 2,
                                  ((*max_pt)[2] + (*min_pt)[2]) / 2);
    // polygon
    if (cloud->size() < 4) {
      obj->polygon.points.resize(4);
      obj->polygon.points[0].x = static_cast<double>((*min_pt)[0]);
      obj->polygon.points[0].y = static_cast<double>((*min_pt)[1]);
      obj->polygon.points[0].z = static_cast<double>((*min_pt)[2]);

      obj->polygon.points[1].x = static_cast<double>((*max_pt)[0]);
      obj->polygon.points[1].y = static_cast<double>((*min_pt)[1]);
      obj->polygon.points[1].z = static_cast<double>((*min_pt)[2]);

      obj->polygon.points[2].x = static_cast<double>((*max_pt)[0]);
      obj->polygon.points[2].y = static_cast<double>((*max_pt)[1]);
      obj->polygon.points[2].z = static_cast<double>((*min_pt)[2]);

      obj->polygon.points[3].x = static_cast<double>((*min_pt)[0]);
      obj->polygon.points[3].y = static_cast<double>((*max_pt)[1]);
      obj->polygon.points[3].z = static_cast<double>((*min_pt)[2]);
    }
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(BaseObjectBuilder);
};

}  // namespace perception
}  // namespace apollo

#endif  // SRC_BASE_OBJECT_BUILDER_H_
