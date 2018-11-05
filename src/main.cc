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

#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "opencv2/opencv.hpp"
#include "pcl/io/pcd_io.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "pcl_types.h"
#include "object.h"

#include "pcl/io/io.h"
#include <pcl/visualization/cloud_viewer.h>
#include <fstream>
#include <unordered_set>
#include "min_box.h"
#include "base_object_builder.h"
#include <math.h>
#include <glob.h>
//#include "fortracking.h"

#define VISUALIZE

using apollo::perception::CNNSegmentation;
using apollo::perception::pcl_util::PointCloud;
using apollo::perception::pcl_util::PointCloudPtr;
using apollo::perception::pcl_util::PointIndices;
using apollo::perception::pcl_util::PointXYZIT;
//using apollo::perception::ObjectBuilderOptions;
//using apollo::perception::MinBoxObjectBuilder;

using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

struct CellStat {
    CellStat() : point_num(0), valid_point_num(0) {}

    int point_num;
    int valid_point_num;
};

int F2I(float val, float ori, float scale) {
    return static_cast<int>(std::floor((ori - val) * scale));
}

cv::Vec3b GetTypeColor(apollo::perception::ObjectType type) {
    switch (type) {
        case apollo::perception::ObjectType::PEDESTRIAN:
            return cv::Vec3b(255, 128, 128);  // pink
        case apollo::perception::ObjectType::BICYCLE:
            return cv::Vec3b(0, 0, 255);  // blue
        case apollo::perception::ObjectType::VEHICLE:
            return cv::Vec3b(0, 255, 0);  // green
        default:
            return cv::Vec3b(0, 255, 255);  // yellow
    }
}

bool IsValidRowCol(int row, int rows, int col, int cols) {
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

int RowCol2Grid(int row, int col, int cols) { return row * cols + col; }

bool GetPointCloudFromFile(const string &pcd_file, PointCloudPtr cloud) {
    pcl::PointCloud<PointXYZIT> ori_cloud;
    if (pcl::io::loadPCDFile(pcd_file, ori_cloud) < 0) {
        std::cerr << "Failed to load pcd file: " << pcd_file;
        return false;
    }

    cloud->points.reserve(ori_cloud.points.size());
    for (size_t i = 0; i < ori_cloud.points.size(); ++i) {
        apollo::perception::pcl_util::Point point;
        point.x = ori_cloud.points[i].x;
        point.y = ori_cloud.points[i].y;
        point.z = ori_cloud.points[i].z;
        point.intensity = ori_cloud.points[i].intensity;
        if (std::isnan(ori_cloud.points[i].x)) {
            continue;
        }
        cloud->push_back(point);
    }

    return true;
}

bool GetPointCloudFromBin(const string &bin_file, PointCloudPtr cloud) {
    //pcl::PointCloud<PointXYZIT> pre_ori_cloud;
    //cloud->points.reserve(pre_ori_cloud.points.size());

    std::ifstream in;
    in.open(bin_file, std::ios::in | std::ios::binary);
    string line;
    struct p {
        float x;
        float y;
        float z;
        float intensity;
    };
    p po;
    int count = 0;
    if (in.is_open()) {
        while (in.read((char *) &po, sizeof(po))) {
            apollo::perception::pcl_util::Point point;
            point.x = po.x;
            point.y = po.y;
            point.z = po.z;
            point.intensity = po.intensity;
            count++;
//            std::cout << " x: " <<  point.x
//                      << " y: " << point.y
//                      << " z: " << point.z
//                      << " intensity: " << point.intensity << std::endl;
            cloud->push_back(point);
        }
        std::cout << "num of points: " << count << std::endl;
    }
    return true;
}

void DrawDetection(PointCloudPtr pc_ptr, const PointIndices &valid_idx,
                   int rows, int cols, float range,
                   const vector<std::shared_ptr<apollo::perception::Object>> &objects,
                   const string &result_file) {
    // create a new image for visualization
    cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(0.0));

    // map points into bird-view grids
    float inv_res_x = 0.5 * static_cast<float>(cols) / range;
    float inv_res_y = 0.5 * static_cast<float>(rows) / range;
    int grids = rows * cols;
    vector<CellStat> view(grids);

    const std::vector<int> *valid_indices_in_pc = &(valid_idx.indices);
    CHECK_LE(valid_indices_in_pc->size(), pc_ptr->size());
    unordered_set<int> unique_indices;
    for (size_t i = 0; i < valid_indices_in_pc->size(); ++i) {
        int point_id = valid_indices_in_pc->at(i);
        CHECK(unique_indices.find(point_id) == unique_indices.end());
        unique_indices.insert(point_id);
    }

    for (size_t i = 0; i < pc_ptr->size(); ++i) {
        const auto &point = pc_ptr->points[i];
        // * the coordinates of x and y have been exchanged in feature generation
        // step,
        // so they should be swapped back here.
        int col = F2I(point.y, range, inv_res_x);  // col
        int row = F2I(point.x, range, inv_res_y);  // row
        if (IsValidRowCol(row, rows, col, cols)) {
            // get grid index and count point number for corresponding node
            int grid = RowCol2Grid(row, col, cols);
            view[grid].point_num++;
            if (unique_indices.find(i) != unique_indices.end()) {
                view[grid].valid_point_num++;
            }
        }
    }

    // show grids with grey color
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int grid = RowCol2Grid(row, col, cols);
            if (view[grid].valid_point_num > 0) {
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(127, 127, 127);
            } else if (view[grid].point_num > 0) {
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(63, 63, 63);
            }
        }
    }

    // show segment grids with tight bounding box
    const cv::Vec3b segm_color(0, 0, 255);  // red

    for (size_t i = 0; i < objects.size(); ++i) {
        const std::shared_ptr<apollo::perception::Object> &obj = objects[i];
        CHECK_GT(obj->cloud->size(), 0);

        int x_min = INT_MAX;
        int y_min = INT_MAX;
        int x_max = INT_MIN;
        int y_max = INT_MIN;
        float score = obj->score;
        CHECK_GE(score, 0.0);
        CHECK_LE(score, 1.0);
        for (size_t j = 0; j < obj->cloud->size(); ++j) {
            const auto &point = obj->cloud->points[j];
            int col = F2I(point.y, range, inv_res_x);  // col
            int row = F2I(point.x, range, inv_res_y);  // row
            CHECK(IsValidRowCol(row, rows, col, cols));
            img.at<cv::Vec3b>(row, col) = segm_color * score;
            x_min = std::min(col, x_min);
            y_min = std::min(row, y_min);
            x_max = std::max(col, x_max);
            y_max = std::max(row, y_max);
        }

        // fillConvexPoly(img, list.data(), list.size(), cv::Scalar(positive_prob *
        // segm_color));
        cv::Vec3b bbox_color = GetTypeColor(obj->type);
//        rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
//                  cv::Scalar(bbox_color));
//    }

        float angle = atan(obj->direction[1] / obj->direction[0]) / CV_PI * 180;
        int width, length;
        if (x_max - x_min < y_max - y_min) {
            width = x_max - x_min;
            length = y_max - y_min;
        } else {
            width = y_max - y_min;
            length = x_max - x_min;
        }
        int center_x = static_cast<int>((x_max + x_min) / 2);
        int center_y = static_cast<int>((y_max + y_min) / 2);
        cv::Size2f size(width, length);
        cv::Point2f center(center_x, center_y);
        cv::RotatedRect cvtr = cv::RotatedRect(center, size, angle);
        cv::Point2f vertics[4];
        cvtr.points(vertics);
        for (size_t i = 0; i < 4; i++) {
            cv::line(img, vertics[i], vertics[(i + 1) % 4], cv::Scalar(bbox_color), 1);
        }
    }

#ifndef VISUALIZE
    // write image intensity values into file
    FILE *f_res;
    f_res = fopen(result_file.c_str(), "w");
    fprintf(f_res, "%d %d\n", rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            fprintf(f_res, "%u %u %u\n", img.at<cv::Vec3b>(row, col)[0],
                    img.at<cv::Vec3b>(row, col)[1], img.at<cv::Vec3b>(row, col)[2]);
        }
    }
    fclose(f_res);
#else
    cv::imwrite(result_file, img);
    std::cout << "save done" << std::endl;
#endif
}

vector<string> globVector(const string &pattern) {
    glob_t glob_result;
    glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    vector<string> files;
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

bool TestCnnSegDet(const string &pcd_file, const string &result_path) {
    // generate input point cloud data
    PointCloudPtr in_pc;
    in_pc.reset(new PointCloud());
    if (!GetPointCloudFromBin(pcd_file, in_pc)) {
        std::cerr << "Failed to load pcd file: " << pcd_file;
    }

    PointIndices valid_idx;
    auto &indices = valid_idx.indices;
    indices.resize(in_pc->size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<std::shared_ptr<apollo::perception::Object>> out_objects;
////    MinBoxObjectBuilder *object_builder_;
////    ObjectBuilderOptions object_builder_options_;
////    object_builder_ = new MinBoxObjectBuilder();
////    object_builder_options_.ref_center = Eigen::Vector3d(0, 0, -1.7);
//    shared_ptr<CNNSegmentation> cnn_segmentor_;
//    cnn_segmentor_.reset(new CNNSegmentation());
//    cnn_segmentor_->Init();
//    cnn_segmentor_->Segment(in_pc, valid_idx, &out_objects);
////    object_builder_->Build(object_builder_options_, &out_objects);
//    DrawDetection(in_pc, valid_idx, cnn_segmentor_->height(),
//                  cnn_segmentor_->width(), cnn_segmentor_->range(), out_objects,
//                  result_path);
//    // cnn_segmentor_->Write2Json(out_objects, json_path);
//    // std::cout << "All process completed" << std::endl;

    // initialization function
    CNNSegmentation cnn_segmentor;
    cnn_segmentor.Init();

//    // segment function
//    for (int i = 0; i < 10; ++i) {
//        cnn_segmentor.Segment(in_pc, valid_idx, &out_objects);
//        // EXPECT_EQ(out_objects.size(), 13);
//    }
    cnn_segmentor.Segment(in_pc, valid_idx, &out_objects);

#ifndef VISUALIZE
    // do visualization of segmentation results (output object detections)
    string result_file(result_path);
    result_file = result_path + "-detection.txt";
    std::cout << result_file << std::endl;
    DrawDetection(in_pc, valid_idx, cnn_segmentor.height(),
                  cnn_segmentor.width(), cnn_segmentor.range(), out_objects,
                  result_file);
#else
//    string result_file(result_path);
//    result_file = result_path + "-detection.png";
//    std::cout << result_file << std::endl;
//    DrawDetection(in_pc, valid_idx, cnn_segmentor.height(),
//                  cnn_segmentor.width(), cnn_segmentor.range(), out_objects,
//                  result_file);
#endif

    return true;
}

int main(int argc, char *argv[]) {
    string path = "/home/zyzhong/git/Apollo-seg/data/*.bin";
    vector<std::string> files;
    files = globVector(path);
    clock_t start, end;
    start = clock();
    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << files[i] << std::endl;
        TestCnnSegDet(files[i], files[i]);
    }
    end = clock();
    printf("Total Time:%f\n",((double)(end - start)/CLOCKS_PER_SEC));
    printf("FPS:%f\n",((double)files.size() * CLOCKS_PER_SEC /(end - start)));

//    Tracking* track = new Tracking();
//    track->Init();
//    track->Process(path, res_png_path);
//    delete track;
    return 0;
}
