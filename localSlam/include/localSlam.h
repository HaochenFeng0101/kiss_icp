#pragma once

#include <memory>
#include <thread>


#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

#include "localSlamTypes.h"
#include "localSlamConfig.h"
// #include "Utils/utils.hpp"
#include "overgrowth_detection.h"

#include <Eigen/Core>

namespace LocalSLAM
{
	class MapOptimiser;
	class LidarOdom;

	
	class LocalSlam
	{
	public:
		LocalSlam(const LocalSlamConfig &config = {});
		~LocalSlam();

		// Reest local slam and start a new session
		void resetSlam();

		// Return true if it is a keyframe
		bool submit(pcl::PointCloud<pcl::PointXYZ>::Ptr points);

		// Return local Slam pointcloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr getLocalMap() const;

		// Return filtered local Slam pointcloud (for object detection e.g overgrown trees)
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getFilteredLocalMap(const std::array<float, 2> length_range,
																   const std::array<float, 2> width_range,
																   const std::array<float, 2> height_range,
																   bool lidar_flip = false) const;

		// get current pose
		Eigen::Matrix4f getCurrentPose() const;
		//get and store preglobalpose
		Eigen::Affine3f get_pre_globalpose();
    	void update_pre_globalpose(Eigen::Affine3f &transform);
		
		//pose matrix
		// std::vector<Eigen::Affine3f> get_pose_matrices() const;
		// void clear_pose_matrices();
		// void add_to_posematrices(Eigen::Affine3f &pose);
			
		
		//timestamped pose
		void save_timestamped_pose_with_time(uint64_t timestamp, Eigen::Affine3f &pose);
		void clear_timestamped_poses();
		std::unordered_map<uint64_t, Eigen::Affine3f> get_timestamped_poses() const;

		//initialize matrix
		void udpate_initial_transform(Eigen::Affine3f &transform);
	    void reset_initial_transform_matrix();
		Eigen::Affine3f get_initial_transform_matrix();
		//initialized bool
		void set_transform_initialized(bool initialized);
		bool get_transform_initialized_bool();
		

	private : LocalSlamConfig m_config;

		std::shared_ptr<MapOptimiser> m_mapOptimiser;
		std::shared_ptr<LidarOdom> m_lidarOdom;

		LocalSLAM::Pose m_latestOdomPose;
		Eigen::Affine3f m_pre_globalpose;

		std::vector<Eigen::Affine3f> m_pose_matrices;
		std::unordered_map<uint64_t, Eigen::Affine3f> m_timestamped_poses;
		std::vector<Eigen::Matrix4f> m_trajectory;
		Eigen::Affine3f m_initial_transform = Eigen::Affine3f::Identity();
		bool m_transform_initialized = false;

		std::vector<LocalSLAM::OverGrowthTreeData> m_overgrown_tree_data;
		// LocalSLAM::OvergrowthDetector m_overgrowth_detect_ptr;

		// filtered segments
		mutable std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> m_filtered_segments;

	};
}
