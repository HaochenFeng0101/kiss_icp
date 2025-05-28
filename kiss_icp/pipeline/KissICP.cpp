// MIT License
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

#include "KissICP.hpp"

#include <Eigen/Core>
#include <tuple>
#include <vector>
#include <iostream>
#include <chrono>

#include "kiss_icp/core/Deskew.hpp"
#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"

namespace kiss_icp::pipeline {

void KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const std::vector<double> &timestamps) {
    const auto &deskew_frame = [&]() -> std::vector<Eigen::Vector3d> {
        if (!config_.deskew) return frame;
        // TODO(Nacho) Add some asserts here to sanitize the timestamps

        //  If not enough poses for the estimation, do not de-skew
        const size_t N = poses().size();
        if (N <= 2) return frame;

        // Estimate linear and angular velocities
        const auto &start_pose = poses_[N - 2];
        const auto &finish_pose = poses_[N - 1];
        return DeSkewScan(frame, timestamps, start_pose, finish_pose);
    }();
    RegisterFrame(deskew_frame);
}

void KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame) {
    // Voxelize
    Voxelize(frame, mDownsampleQueryBuffer, mDownsampleMapBuffer);
    // Uniform downsampling
    UniformSample(mDownsampleQueryBuffer, mDownsampleQueryBuffer);
    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    const auto prediction = GetPredictionModel();
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    Sophus::SE3d initial_guess = last_pose * prediction;

    // Run icp
    const Sophus::SE3d new_pose = kiss_icp::RegisterFrame(mDownsampleQueryBuffer,         //
                                                          local_map_,     //
                                                          initial_guess,  //
                                                          3.0 * sigma,    //
                                                          sigma / 3.0);

    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(mDownsampleMapBuffer, new_pose);
    poses_.push_back(new_pose);
}

void KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const Eigen::Vector3f& prior_pos, const Eigen::Quaternionf& prior_orien) {
    // Voxelize
    Voxelize(frame, mDownsampleQueryBuffer, mDownsampleMapBuffer);
    // Uniform downsampling
    UniformSample(mDownsampleQueryBuffer, mDownsampleQueryBuffer);
    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    // Set pose prior
    Sophus::SE3d initial_guess;
    initial_guess.translation() = prior_pos.cast<double>();
    initial_guess.setQuaternion(prior_orien.cast<double>());

    // Run icp
    const Sophus::SE3d new_pose = kiss_icp::RegisterFrame(mDownsampleQueryBuffer,         //
                                                          local_map_,     //
                                                          initial_guess,  //
                                                          3.0 * sigma,    //
                                                          sigma / 3.0);

    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(mDownsampleMapBuffer, new_pose);
    poses_.push_back(new_pose);
}

void KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame, const Eigen::Quaternionf& orientation_prior) {
    // Voxelize
    Voxelize(frame, mDownsampleQueryBuffer, mDownsampleMapBuffer);
    // Uniform downsampling
    UniformSample(mDownsampleQueryBuffer, mDownsampleQueryBuffer);
    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    // Start with KissICP's initial guess, then replace the orientation
    const auto prediction = GetPredictionModel();
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3d();
    Sophus::SE3d initial_guess = last_pose * prediction;
    initial_guess.setQuaternion(orientation_prior.cast<double>());

    // Run icp
    const Sophus::SE3d new_pose = kiss_icp::RegisterFrame(mDownsampleQueryBuffer,         //
                                                          local_map_,     //
                                                          initial_guess,  //
                                                          3.0 * sigma,    //
                                                          sigma / 3.0);

    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(mDownsampleMapBuffer, new_pose);
    poses_.push_back(new_pose);
}

void KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame, std::vector<Eigen::Vector3d>& query_buffer, std::vector<Eigen::Vector3d>& map_buffer) const {
    const auto voxel_size = config_.voxel_size;
    map_buffer = kiss_icp::VoxelDownsample(frame, voxel_size * 0.5);
    query_buffer = kiss_icp::VoxelDownsample(map_buffer, voxel_size * 1.5);
}

double KissICP::GetAdaptiveThreshold() {
    if (!HasMoved()) {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

Sophus::SE3d KissICP::GetPredictionModel() const {
    Sophus::SE3d pred = Sophus::SE3d();
    const size_t N = poses_.size();
    if (N < 2) return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool KissICP::HasMoved() {
    if (poses_.empty()) return false;
    const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
    return motion > 5.0 * config_.min_motion_th;
}

void KissICP::clearPoses() {
    local_map_.Clear();
    poses_.clear();
}

void KissICP::UniformSample(const std::vector<Eigen::Vector3d> &frameIn, std::vector<Eigen::Vector3d> &frameOut){
    frameOut = frameIn;
    if(static_cast<int>(frameOut.size()) <= config_.max_downsampled_points){ return; }

    std::random_device rd;
    std::mt19937 mer_twister(rd());
    std::shuffle(frameOut.begin(), frameOut.end(), mer_twister);
    frameOut.resize(config_.max_downsampled_points);
}

void KissICP::SetMapWithVoxelize(const std::vector<Eigen::Vector3d>& map)
{
    const auto dsMap = kiss_icp::VoxelDownsample(map, config_.voxel_size * 0.5);
    SetMap(dsMap);
}

void KissICP::SetMap(const std::vector<Eigen::Vector3d>& map){
    local_map_.Clear();
    local_map_.AddPoints(map);
}

void KissICP::AddTransform(const Eigen::Vector3d& pose, const Eigen::Quaterniond& quat)
{
    Sophus::SE3d tf;
    tf.translation() = pose;
    tf.setQuaternion(quat);
    poses_.push_back(tf);
}

std::pair<Sophus::SE3d, double> KissICP::FindFrameInMap(const std::vector<Eigen::Vector3d> &frame, const Eigen::Vector3f& prior_pos, const Eigen::Quaternionf& prior_orien){
    // Voxelize
    Voxelize(frame, mDownsampleQueryBuffer, mDownsampleMapBuffer);

    // Uniform downsampling
    UniformSample(mDownsampleQueryBuffer, mDownsampleQueryBuffer);
    
    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3d initial_guess;
    initial_guess = Sophus::SE3d();
    initial_guess.translation() = prior_pos.cast<double>();
    initial_guess.setQuaternion(prior_orien.cast<double>());

    // Run icp with cost query
    double cost;
    auto new_pose = kiss_icp::RegisterFrame(mDownsampleQueryBuffer, //
                                            local_map_,             //
                                            initial_guess,          //
                                            3.0 * sigma,            //
                                            sigma / 3.0,            //
                                            cost);                  //


    return std::make_pair(new_pose, cost);
}

}  // namespace kiss_icp::pipeline
