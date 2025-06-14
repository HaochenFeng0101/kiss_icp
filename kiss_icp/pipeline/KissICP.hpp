//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <Eigen/Core>
#include <Eigen/src/Geometry/Quaternion.h>
#include <tuple>
#include <vector>
#include <optional>
#include <algorithm>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Threshold.hpp"
#include "kiss_icp/core/VoxelHashAndAlign.hpp"

namespace kiss_icp::pipeline {

struct KISSConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;
    int max_downsampled_points = 1500;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    // Motion compensation
    bool deskew = false;
};

class KissICP {
public:
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

public:
    explicit KissICP(const KISSConfig &config)
        : config_(config),
          local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel),
          adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range) {
            mDownsampleQueryBuffer.reserve(65535);
            mDownsampleMapBuffer.reserve(65535);
          }

    KissICP() : KissICP(KISSConfig{}) {}

public:
    // Used to give no prior
    void RegisterFrame(const std::vector<Eigen::Vector3d> &frame);
    // Used to give full pose prior
    // void RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const dashTypes::Pose& prior);
    void RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const Eigen::Vector3f& prior_pos, const Eigen::Quaternionf& prior_orien);
    // Used for only orientation prior only
    void RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const Eigen::Quaternionf& orientation_prior);
    // Used if u need KissICP's internal deskewing (untested)
    void RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const std::vector<double> &timestamps);
    void Voxelize(const std::vector<Eigen::Vector3d> &frame, std::vector<Eigen::Vector3d>& query_buffer, std::vector<Eigen::Vector3d>& map_buffer) const;
    double GetAdaptiveThreshold();
    Sophus::SE3d GetPredictionModel() const;
    void UniformSample(const std::vector<Eigen::Vector3d> &frameIn, std::vector<Eigen::Vector3d> &frameOut);
    bool HasMoved();

    // reset kiss-ICP
    void clearPoses();
    void AddTransform(const Eigen::Vector3d& pose, const Eigen::Quaterniond& quat);

public: // Scan-To-Map Mode
    void SetMap(const std::vector<Eigen::Vector3d>& map);
    void SetMapWithVoxelize(const std::vector<Eigen::Vector3d>& map);
    // std::pair<Sophus::SE3d, double> FindFrameInMap(const std::vector<Eigen::Vector3d> &frame, const dashTypes::Pose& prior);
    std::pair<Sophus::SE3d, double> FindFrameInMap(const std::vector<Eigen::Vector3d> &frame, const Eigen::Vector3f& prior_pos, const Eigen::Quaternionf& prior_orien);

public:
    // Extra C++ API to facilitate ROS debugging
    std::vector<Eigen::Vector3d> LocalMap() const { return local_map_.Pointcloud(); };
    std::vector<Sophus::SE3d> poses() const { return poses_; };

private:
    // KISS-ICP pipeline modules
    std::vector<Sophus::SE3d> poses_;
    KISSConfig config_;
    dashICP::VoxelHashAndAlign local_map_;
    AdaptiveThreshold adaptive_threshold_;

    std::vector<Eigen::Vector3d> mDownsampleQueryBuffer;
    std::vector<Eigen::Vector3d> mDownsampleMapBuffer;
};

}  // namespace kiss_icp::pipeline
