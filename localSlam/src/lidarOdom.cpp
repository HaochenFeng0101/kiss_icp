
#include "lidarOdom.h"
#include <spdlog/spdlog.h>

namespace LocalSLAM
{	
	

	LidarOdom::LidarOdom(const LidarOdomConfig& configIn)
	{
		kiss_icp::pipeline::KISSConfig kicp_config = m_config.kiss_icp_config;
		m_kissICP                                  = std::make_shared<kiss_icp::pipeline::KissICP>(kicp_config);

		// Init downsampler and buffer
		m_voxelDownsampler.setLeafSize(0.05f, 0.05f, 0.05f);
		auto range = m_config.kiss_icp_config.max_range;
		m_farCropBox.setMin(Eigen::Vector4f(-range, -range, -range, 1.0f));
		m_farCropBox.setMax(Eigen::Vector4f(range, range, range, 1.0f));
		// m_downsampleBuffer = DeskewedPointCloud::Initialised();
		m_downsampleBuffer.reset(new pcl::PointCloud<pcl::PointXYZ>);
		spdlog::info("[KissICP Lidar Odom] - Constructed!");
	}

	LidarOdom::~LidarOdom() = default;

		LocalSLAM::Pose LidarOdom::getLidarOdomPose(
			pcl::PointCloud<pcl::PointXYZ>::Ptr       pclIn,
			const std::optional<LocalSLAM::Pose>& priorPose
		)
		{
			// Far Crop and Voxel Downsample
			m_farCropBox.setInputCloud(pclIn);
			m_farCropBox.filter(*m_downsampleBuffer);
			m_voxelDownsampler.setInputCloud(m_downsampleBuffer);
			m_voxelDownsampler.filter(*(m_downsampleBuffer));
			// Point Skipping
			// PointSkip(m_downsampleBuffer, m_config.max_icp_points_registered);
			// auto start_time = std::chrono::system_clock::now();

			// Convert to Eigen pointcloud
			int num_pt = m_downsampleBuffer->size();
		// std::cout << num_pt << std::endl;
		m_pcBuffer.resize(num_pt);
		for (int i = 0; i < num_pt; i++)
		{
			m_pcBuffer[i] = (Eigen::Vector3d(
				m_downsampleBuffer->points[i].x,
				m_downsampleBuffer->points[i].y,
				m_downsampleBuffer->points[i].z
			

			));
		}

		// Give only IMU orientation in prior
		if (priorPose.has_value())
		{
			m_kissICP->RegisterFrame(m_pcBuffer, priorPose.value().position, priorPose.value().orientation);
		}
		else
		{
			m_kissICP->RegisterFrame(m_pcBuffer);
		}

		auto           kiss_icp_result = m_kissICP->poses().back(); // Can't reference, poses() return a copy
		LocalSLAM::Pose resultOut;
		resultOut.position    = kiss_icp_result.translation().cast<float>();
		resultOut.orientation = kiss_icp_result.unit_quaternion().cast<float>();

		// std::cout << "Lidar Odom Completed in "<< std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count()<< " ms" << std::endl;

		return resultOut;
	}

	void LidarOdom::clearPoses()
	{
		if (m_kissICP){
			m_kissICP->clearPoses();
		}
		m_downsampleBuffer.reset(new pcl::PointCloud<pcl::PointXYZ>);
		// m_downsampleBuffer = DeskewedPointCloud::Initialised();
	}

	// void LidarOdom::PointSkip(pcl::PointCloud<pcl::PointXYZ>::Ptr pclIn, const int& target_count)
	// 	//DeskewedPointCloud & pclIn, const int& target_count)
	// {
	// 	int num_pts = pclIn->pointcloud->size();
	// 	if (num_pts <= target_count)
	// 	{
	// 		return;
	// 	}
	// 	float interval = float(num_pts - 1) / float(target_count - 1);
	// 	for (int i = 0; i < target_count; i++)
	// 	{
	// 		pclIn.pointcloud->at(i) = pclIn.pointcloud->at(std::floor(i * interval));
	// 	}
	// 	// pclIn.pointcloud->resize(target_count);
	// }

}

