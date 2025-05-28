#include "VoxelHashAndAlign.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <iostream>

#include <Eigen/Core>
#include <algorithm>
#include <mutex>
#include <limits>
#include <tuple>
#include <utility>

// This parameters are not intended to be changed, therefore we do not expose it


namespace Eigen {
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {
    static constexpr int MAX_COMPUTE_THREADS = 5;

    struct ResultTuple {
        ResultTuple(std::size_t n) {
            source.reserve(n);
            target.reserve(n);
        }
        std::vector<Eigen::Vector3d> source;
        std::vector<Eigen::Vector3d> target;
    };

    inline double square(double x) { return x * x; }

    struct RegisterTuple {
        RegisterTuple() {
            JTJ.setZero();
            JTr.setZero();
        }

        RegisterTuple operator+(const RegisterTuple& other) {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        Eigen::Matrix6d JTJ;
        Eigen::Vector6d JTr;
    };

    static std::array<RegisterTuple, MAX_COMPUTE_THREADS> registerTuplesList;
    static std::array<bool, MAX_COMPUTE_THREADS> registerTuplesActiveThreads;
}  // namespace

namespace dashICP {

VoxelHashAndAlign::Vector3dVectorTuple VoxelHashAndAlign::GetCorrespondences(
    const Vector3dVector& points, double max_correspondance_distance) const {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighboor = [&](const Eigen::Vector3d& point) {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighboors;
        neighboors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto& voxel) {
            auto search = map_.find(voxel);
            if (search != map_.end()) {
                const auto& points = search->second.points;
                if (!points.empty()) {
                    for (const auto& point : points) {
                        neighboors.emplace_back(point);
                    }
                }
            }
            });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        std::for_each(neighboors.cbegin(), neighboors.cend(), [&](const auto& neighbor) {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
            });

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighboor](
            const tbb::blocked_range<points_iterator>& r, ResultTuple res) -> ResultTuple {
                auto& [src, tgt] = res;
                src.reserve(r.size());
                tgt.reserve(r.size());
                for (const auto& point : r) {
                    Eigen::Vector3d closest_neighboors = GetClosestNeighboor(point);
                    if ((closest_neighboors - point).norm() < max_correspondance_distance) {
                        src.emplace_back(point);
                        tgt.emplace_back(closest_neighboors);
                    }
                }
                return res;
        },
        // 2nd lambda: Parallel reduction
            [](ResultTuple a, const ResultTuple& b) -> ResultTuple {
            auto& [src, tgt] = a;
            const auto& [srcp, tgtp] = b;
            src.insert(src.end(),  //
                std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

VoxelHashAndAlign::Vector3dVectorTuple VoxelHashAndAlign::GetCorrespondencesFused(const Vector3dVector& points, double max_correspondance_distance) const
{
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, this](
            const tbb::blocked_range<points_iterator>& r, ResultTuple res) -> ResultTuple {
                auto& [src, tgt] = res;
                src.reserve(r.size());
                tgt.reserve(r.size());
                for (const auto& point : r) {
                    std::pair<Eigen::Vector3d, double> neighborsInfo = GetClosestNeighborWithDist2(point);
                    Eigen::Vector3d closest_neighboors = neighborsInfo.first;
                    if ((closest_neighboors - point).norm() < max_correspondance_distance) {
                        src.emplace_back(point);
                        tgt.emplace_back(closest_neighboors);
                    }
                }
                return res;
        },
        // 2nd lambda: Parallel reduction
            [](ResultTuple a, const ResultTuple& b) -> ResultTuple {
            auto& [src, tgt] = a;
            const auto& [srcp, tgtp] = b;
            src.insert(src.end(),  //
                std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

Sophus::SE3d
VoxelHashAndAlign::GetCorrespondencesAndAlign(
    const Vector3dVector& points, 
    double max_correspondance_distance,
    double th) const
{
    auto compute_jacobian = [&](const Eigen::Vector3d& sourcePt) 
    {
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -Sophus::SO3d::hat(sourcePt);
        return J_r;
    };

    std::fill(registerTuplesList.begin(), registerTuplesList.end(), RegisterTuple());
    // Parallel compute for individual Jacobian and residuals for each thread partition
    std::fill(registerTuplesActiveThreads.begin(), registerTuplesActiveThreads.end(), false);

    auto processPoints = [&, this]() -> void
    {
        auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };
        
        #pragma omp parallel for num_threads(MAX_COMPUTE_THREADS)
        for(int i = 0; i < static_cast<int>(points.size()); i++) {
            const Eigen::Vector3d& cPt = points[i];
            std::tuple<Eigen::Vector3d, double, Eigen::Vector3d> neighborsInfo = GetClosestNeighborWithDist2Residual(cPt);
            double closest_dist2 = std::get<1>(neighborsInfo);
            Eigen::Vector3d residual = std::get<2>(neighborsInfo);

            if (closest_dist2 < square(max_correspondance_distance)) {
                int thread_idx = omp_get_thread_num();
                registerTuplesActiveThreads[thread_idx] = true;

                RegisterTuple& cRegisterTuple = registerTuplesList.at(thread_idx);
                Eigen::Matrix6d& JTJ_private = cRegisterTuple.JTJ;
                Eigen::Vector6d& JTr_private = cRegisterTuple.JTr;
                Eigen::Matrix3_6d J_r = compute_jacobian(cPt);
                const double w = Weight(closest_dist2);
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
        }
    };
   
    processPoints();

    // Sum up/gather all the terms into the final JTJ and JTr components
    RegisterTuple finalJTvals;
    for (size_t i = 0; i < registerTuplesActiveThreads.size(); i++) {
        bool partitionActive = registerTuplesActiveThreads.at(i);
        if (partitionActive) {
            RegisterTuple& cRegisterTuple = registerTuplesList.at(i);
            finalJTvals = finalJTvals + cRegisterTuple;
        }
    }

    // Now perform Cholesky decomposition
    const Eigen::Vector6d x = finalJTvals.JTJ.ldlt().solve(-finalJTvals.JTr);
    return Sophus::SE3d::exp(x);
}

std::pair<Eigen::Vector3d, double> VoxelHashAndAlign::GetClosestNeighborWithDist2(const Eigen::Vector3d& point) const
{
    Eigen::Vector3d closest_neighbor;
    double closest_distance2 = std::numeric_limits<double>::max();
    auto kx = static_cast<int>(point[0] / voxel_size_);
    auto ky = static_cast<int>(point[1] / voxel_size_);
    auto kz = static_cast<int>(point[2] / voxel_size_);

    for (int i = kx - 1; i < kx + 1 + 1; ++i) {
        for (int j = ky - 1; j < ky + 1 + 1; ++j) {
            for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                // Try to find the voxel in the entire 3D grid map
                Voxel cVoxel(i, j, k);
                auto search = map_.find(cVoxel);
                if (search != map_.end()) {
                    const auto& points = search->second.points;
                    if (!points.empty()) {
                        // Go through each neighbor point and find the closest
                        for (const Eigen::Vector3d& neighborPt : points) {
                            double distance = (neighborPt - point).squaredNorm();
                            if (distance < closest_distance2) {
                                closest_neighbor = neighborPt;
                                closest_distance2 = distance;
                                if(closest_distance2 < 0.05*0.05){
                                    return std::make_pair(closest_neighbor, closest_distance2);
                                }
                            }
                        }
                        // End of search for closest neighbor point
                    }
                }
                // End of finding voxel in 3D grid map
            }
        }
    }

    return std::make_pair(closest_neighbor, closest_distance2);
}

std::tuple<Eigen::Vector3d, double, Eigen::Vector3d> 
VoxelHashAndAlign::GetClosestNeighborWithDist2Residual(const Eigen::Vector3d& point) const
{
    Eigen::Vector3d closest_neighbor;
    double closest_distance2 = std::numeric_limits<double>::max();
    Eigen::Vector3d residual;
    double recpVoxelSize = 1.0 / voxel_size_;
    Eigen::Vector3d kVec = point * recpVoxelSize;

    auto kx = static_cast<int>(kVec[0]);
    auto ky = static_cast<int>(kVec[1]);
    auto kz = static_cast<int>(kVec[2]);

    for (int i = kx - 1; i < kx + 1 + 1; ++i) {
        for (int j = ky - 1; j < ky + 1 + 1; ++j) {
            for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                // Try to find the voxel in the entire 3D grid map
                Voxel cVoxel(i, j, k);
                auto search = map_.find(cVoxel);
                if (search != map_.end()) {
                    const auto& points = search->second.points;
                    // Go through each neighbor point and find the closest
                    for (const Eigen::Vector3d& neighborPt : points) {
                        Eigen::Vector3d cResidual = (point - neighborPt);
                        double distance = cResidual.squaredNorm();
                        if (distance < closest_distance2) {
                            closest_neighbor = neighborPt;
                            closest_distance2 = distance;
                            residual = cResidual;
                        }
                    }
                    // End of search for closest neighbor point
                }
                // End of finding voxel in 3D grid map
            }
        }
    }

    return std::tuple(closest_neighbor, closest_distance2, residual);
}

std::vector<Eigen::Vector3d> VoxelHashAndAlign::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto& [voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto& point : voxel_block.points) {
            points.push_back(point);
        }
    }
    return points;
}

void VoxelHashAndAlign::Update(const Vector3dVector& points, const Eigen::Vector3d& origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashAndAlign::Update(const Vector3dVector& points, const Sophus::SE3d& pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
        [&](const auto& point) { return pose * point; });
    const Eigen::Vector3d& origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashAndAlign::AddPoints(const std::vector<Eigen::Vector3d>& points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto& voxel_block = search.value();
            voxel_block.AddPoint(point);
        }
        else {
            map_.insert({ voxel, VoxelBlock{{point}, max_points_per_voxel_} });
        }
        });
}

void VoxelHashAndAlign::RemovePointsFarFromLocation(const Eigen::Vector3d& origin) {
    const auto max_distance2 = max_distance_ * max_distance_;
    for (const auto& [voxel, voxel_block] : map_) {
        const auto& pt = voxel_block.points.front();
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            map_.erase(voxel);
        }
    }
}

double VoxelHashAndAlign::ComputeCost(const Vector3dVector& points, double max_correspondance_distance, double th) const{
    auto Weight = [&](double residual2) { return square(th) / square(th + residual2); };

    double total_cost = 0;
    for(int i = 0; i < static_cast<int>(points.size()); i++) {
        const Eigen::Vector3d& cPt = points[i];
        std::tuple<Eigen::Vector3d, double, Eigen::Vector3d> neighborsInfo = GetClosestNeighborWithDist2Residual(cPt);
        double closest_dist2 = std::get<1>(neighborsInfo);

        if (closest_dist2 < square(max_correspondance_distance)) {
            const double curr_cost = Weight(closest_dist2);
            total_cost += curr_cost;
        }
    }

    return total_cost;
}

}  // namespace