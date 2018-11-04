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

#include "fortracking.h"

void Tracking::Init()
{
    cnn_segmentor_.reset(new CNNSegmentation());
    cnn_segmentor_->Init();
    object_builder_.reset(new MinBoxObjectBuilder());
    object_builder_options_.ref_center = Eigen::Vector3d(0, 0, -1.7);
    files.clear();
//    in_pc.reset(new PointCloud);
}

void Tracking::globfiles(const string& pattern)
{
        glob_t glob_result;
        glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
        for(unsigned int i=0;i<glob_result.gl_pathc;++i)
        {
            files.push_back(string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
}

bool Tracking::GetPointCloudFromFile(const string &pcd_file, PointCloudPtr& cloud)
{
    pcl::PointCloud<PointXYZIT> pre_ori_cloud;
    if (pcl::io::loadPCDFile(pcd_file, pre_ori_cloud) < 0)
    {
        AERROR << "Failed to load pcd file: " << pcd_file;
        return false;
    }
    cloud->points.reserve(pre_ori_cloud.points.size());
    for(size_t i = 0; i < pre_ori_cloud.points.size(); ++i)
    {
        apollo::perception::pcl_util::Point point;
        point.x = pre_ori_cloud.points[i].x;
        point.y = pre_ori_cloud.points[i].y;
        point.z = pre_ori_cloud.points[i].z;
        point.intensity = pre_ori_cloud.points[i].intensity;
        if (std::isnan(pre_ori_cloud.points[i].x)) {
            continue;
        }
        cloud->push_back(point);
    }
    std::cout << "pcl read: " << cloud->size() << std::endl;
    return true;
}

bool Tracking::GetPointCloudFromBin(const string &bin_file, PointCloudPtr& cloud)
{
    std::ifstream in;
    in.open(bin_file, std::ios::in | std::ios::binary);
    string line;
    struct p{
        float x;
        float y;
        float z;
        float intensity;
    };
    p po;
    int count = 0;
    if(in.is_open()) {
        while (in.read((char *) &po, sizeof(po))) {
            apollo::perception::pcl_util::Point point;
            point.x = po.x;
            point.y = po.y;
            point.z = po.z;
            point.intensity = po.intensity;
            count++;
            std::cout << "x: "<<  point.x << " y: " << point.y << " z: " << point.z << " intensity: " << point.intensity << std::endl;
            cloud->push_back(point);
        }
        std::cout << "num of points: " << count << std::endl;
    }
    return true;
}

void Tracking::DrawDetection(const PointCloudPtr &pc_ptr, const PointIndices &valid_idx,
                   int rows, int cols, float range,
                   const vector<ObjectPtr> &objects,
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
    std::unordered_set<int> unique_indices;
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
        const ObjectPtr &obj = objects[i];
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

        cv::Vec3b bbox_color = GetTypeColor(obj->type);

        float angle = static_cast<float>(atan(obj->direction[1] / obj->direction[0]) / CV_PI * 180);
        int width, length;
        if (x_max-x_min < y_max - y_min)
        {
            width = x_max - x_min;
            length = y_max - y_min;
        }
        else
        {
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
        for(size_t i=0; i<4;i++)
        {
            cv::line(img, vertics[i], vertics[(i+1)%4], cv::Scalar(bbox_color),1);
        }
    }
    if (!cv::imwrite(result_file, img)) {
        return;
    }
    std::cout << "save done " << std::endl;
}

void Tracking::pre_tracking(string& point_path, string& png_path)
{
    PointCloudPtr in_pc(new PointCloud);
    PointIndices valid_idx;
    auto &indices = valid_idx.indices;
    GetPointCloudFromBin(point_path, in_pc);
    indices.resize(in_pc->size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<ObjectPtr> out_objects;
    cnn_segmentor_->Segment(in_pc, valid_idx, &out_objects);
//    std::cout << "begin out_objects: " << in_pc->size() << std::endl;
    object_builder_->Build(object_builder_options_, &out_objects);
    DrawDetection(in_pc, valid_idx, cnn_segmentor_->height(),
                  cnn_segmentor_->width(), cnn_segmentor_->range(), out_objects,
                  png_path);

}

void Tracking::Process(string& dir, string& png_path)
{
    globfiles(dir);
    for(int i=0; i<files.size(); ++i)
    {
        std::cout << files[i] << std::endl;
        string finial = png_path + std::to_string(i) + ".png";
        pre_tracking(files[i], finial);
        std::cout << "-----------------------" << std::endl;
        std::cout << "current " << i << " file" << std::endl;
        std::cout << "-----------------------" << std::endl;
    }
//    std::string path = "/home/bai/kitti/bin/003485.bin";
//    string finial = "/home/bai/kitti/res/" + std::to_string(3485) + ".png";
//    pre_tracking(path, finial);
}