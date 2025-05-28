#pragma once

#include <vector>
#include <shared_mutex>
#include "localSlamTypes.h"
#include "localSlamConfig.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>


namespace LocalSLAM
{
	class MapOptimiser
	{
	public:
		MapOptimiser(const MapOptimiserConfig& config = {});

		bool updatePoseGraphWithKeyframes(pcl::PointCloud<pcl::PointXYZ>::Ptr pcPtr, Eigen::Vector3f position, Eigen::Quaternionf rotation, uint64_t timestamp);

		// This returns mutable pointcloud!! CANNOT modify the pointcloud ptr returned here
		// If you need to modify it, please clone it
		pcl::PointCloud<pcl::PointXYZ>::Ptr const getPointCloudPtrAtFrame(int frameID) const;


		// For debugging, will remove in the future
		std::vector<Eigen::Vector3f> getPointcloudEigenVector3f() const;
		std::vector<Eigen::Vector3d> getPointcloudEigenVector3d() const;
		pcl::PointCloud<pcl::PointXYZ>::Ptr getPCLPointcloud() const;
		std::vector<PoseData> const& getPoses() const;
		std::vector<PoseData> getKeyframePosesCpy() const;
		Eigen::Affine3f getKeyframeAffine3f(size_t keyframeID) const;
		size_t getNumKeyframes() const;

	private:
		bool checkShouldCache() const;

		void addOdomFactor();

		static inline Eigen::Affine3f poseDataToAffine3f(const PoseData& thisPoint);

		static inline gtsam::Pose3 poseDataToGtsamPose3(const PoseData& thisPoint);
		static inline gtsam::Pose3 trans2gtsamPose(float transformIn[]);
		static inline gtsam::Pose3 eigenAffine2gtsamPose(const Eigen::Affine3f& affined);

		void updateAllPoses();

		// This perform a transformation correction update
		// Hence, avoid rebuilding the entire map for Lidar Odom
		void updateWorldTOdomTransform(const Eigen::Affine3f& odomWorldTscan);

	private:
		MapOptimiserConfig m_config;
		float m_transformToBeMapped[6];
		float m_maxKeyframeDistSquare{0.0f};

		mutable std::shared_mutex m_keyPoseMtx;
		std::vector<PoseData> m_keyPoseWithTimestamp;
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> m_keyframePointclouds;

		// Variabled used in algorithm
		uint64_t m_currentTimestamp;
		Eigen::Affine3f mWorldTOdomWorld = Eigen::Affine3f::Identity(); // Use to correct odometry data to the corrected world frame (especially with loop closure poses changes)

		// gtsam
		gtsam::ISAM2                m_mapOptimiser;
		gtsam::NonlinearFactorGraph m_mapGraph;
		gtsam::Values               m_mapInitGraphValues;
		gtsam::Values               m_mapCurrGraphValues;
	};			
}
