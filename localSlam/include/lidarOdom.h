#pragma once

#include "localSlamConfig.h"
#include "localSlamTypes.h"
#include "KissICP.hpp"
#include <memory>
#include <optional>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include "kiss_icp/core/Deskew.hpp" 
#include <localSlamTypes.h>

namespace LocalSLAM{
	
	
	class LidarOdom{
		public:
			LidarOdom(const LidarOdomConfig& configIn);
			~LidarOdom();

			
			LocalSLAM::Pose getLidarOdomPose(pcl::PointCloud<pcl::PointXYZ>::Ptr pclIn, const std::optional<LocalSLAM::Pose>& priorPose = std::nullopt);
			
			void clearPoses();
			
		private:
			// void PointSkip(pcl::PointCloud<pcl::PointXYZ>::Ptr pclIn, const int& target_count);

		private:
			LidarOdomConfig m_config;
			std::shared_ptr<kiss_icp::pipeline::KissICP> m_kissICP;
			std::vector<Eigen::Vector3d> m_pcBuffer;
			pcl::VoxelGrid<pcl::PointXYZ> m_voxelDownsampler;
			// DeskewedPointCloud m_downsampleBuffer;
			pcl::PointCloud<pcl::PointXYZ>::Ptr m_downsampleBuffer;
			pcl::CropBox<pcl::PointXYZ> m_farCropBox;
	};


}