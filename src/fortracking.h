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

#ifndef SRC_FORTRACKING_H
#define SRC_FORTRACKING_H

#include <iostream>
#include <vector>
#include <string>
#include <glob.h>
#include "min_box.h"
#include "base_object_builder.h"
#include "cluster2d.h"
#include "pcl_types.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include "cnn_segmentation.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <unordered_set>

using std::string;
using std::vector;
using apollo::perception::CNNSegmentation;
using apollo::perception::ObjectPtr;
using apollo::perception::pcl_util::PointCloud;
using apollo::perception::pcl_util::PointCloudPtr;
using apollo::perception::pcl_util::PointIndices;
using apollo::perception::pcl_util::PointXYZIT;
using apollo::perception::ObjectType;
using apollo::perception::ObjectBuilderOptions;
using apollo::perception::MinBoxObjectBuilder;
using std::shared_ptr;

class Tracking
{
    public:
        Tracking() {};
        ~Tracking() {};
        void Init();
        void globfiles(const string& pattern); //get files from one dirs, using glob func
        bool GetPointCloudFromFile(const string &pcd_file, PointCloudPtr& cloud);   //from pcd
        bool GetPointCloudFromBin(const string &bin_file, PointCloudPtr& cloud);  //from bin
        void DrawDetection(const PointCloudPtr &pc_ptr, const PointIndices &valid_idx, \
                           int rows, int cols, float range,  \
                           const vector<ObjectPtr> &objects,  \
                           const string &result_file);
        void pre_tracking(string& point_path, string& png_path);
        cv::Vec3b GetTypeColor(ObjectType type)
        {
            switch (type) {
                case ObjectType::PEDESTRIAN:
                    return cv::Vec3b(255, 128, 128);  // pink
                case ObjectType::BICYCLE:
                    return cv::Vec3b(0, 0, 255);  // blue
                case ObjectType::VEHICLE:
                    return cv::Vec3b(0, 255, 0);  // green
                default:
                    return cv::Vec3b(0, 255, 255);  // yellow
            }
        }

        void Process(string& dir, string& png_path);


    private:

        vector<string> files;
        std::unique_ptr<CNNSegmentation> cnn_segmentor_;
        std::unique_ptr<MinBoxObjectBuilder> object_builder_;
        ObjectBuilderOptions object_builder_options_;

        int F2I(float val, float ori, float scale)
        {
            return static_cast<int>(std::floor((ori - val) * scale));
        }

        bool IsValidRowCol(int row, int rows, int col, int cols)
        {
            return row >= 0 && row < rows && col >= 0 && col < cols;
        }

        int RowCol2Grid(int row, int col, int cols)
        {
            return row * cols + col;
        }
        struct CellStat
        {
            CellStat() : point_num(0), valid_point_num(0) {}
            int point_num;
            int valid_point_num;
        };


};

#endif //SRC_FORTRACKING_H