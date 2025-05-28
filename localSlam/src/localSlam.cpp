#include "localSlam.h"

#include "lidarOdom.h"
#include "mapOptimiser.h"

#include <Eigen/src/Geometry/Transform.h>
#include <spdlog/spdlog.h>

namespace LocalSLAM
{
	LocalSlam::LocalSlam(const LocalSlamConfig &config)
		: m_config(config), m_mapOptimiser(std::make_shared<MapOptimiser>(config.mapOptimConfig)), m_lidarOdom(std::make_shared<LidarOdom>(config.lidarOdomConfig)), m_latestOdomPose{}
	{
	}

	LocalSlam::~LocalSlam()
	{
		//delete 
	}

	void LocalSlam::resetSlam()
	{
		m_lidarOdom->clearPoses();
		m_mapOptimiser.reset();
		m_mapOptimiser = std::make_shared<MapOptimiser>(m_config.mapOptimConfig);
		m_trajectory.clear(); // Clear trajectory when resetting
		m_initial_transform = Eigen::Affine3f::Identity();
		m_transform_initialized = false;
		m_timestamped_poses.clear();
		m_pre_globalpose = Eigen::Affine3f::Identity();
		spdlog::info("{} - Local Slam Reset!", __func__);
	}

	bool LocalSlam::submit(pcl::PointCloud<pcl::PointXYZ>::Ptr points)
	{
		if (!m_lidarOdom || !m_mapOptimiser)
		{
			return false;
		}
		m_latestOdomPose = m_lidarOdom->getLidarOdomPose(points);
		m_latestOdomPose.orientation.normalize();

		// Store current pose in trajectory
		Eigen::Matrix4f current_pose = getCurrentPose();
		m_trajectory.push_back(current_pose);

		return m_mapOptimiser->updatePoseGraphWithKeyframes(
			points,
			m_latestOdomPose.position,
			m_latestOdomPose.orientation,
			points->header.stamp);
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr LocalSlam::getLocalMap() const
	{
		if (m_mapOptimiser)
		{
			return m_mapOptimiser->getPCLPointcloud();
		}
		spdlog::warn("{} - No map optimiser exists", __func__);
		return pcl::PointCloud<pcl::PointXYZ>::Ptr{};
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr LocalSlam::getFilteredLocalMap(const std::array<float, 2> length_range,
																		  const std::array<float, 2> width_range,
																		  const std::array<float, 2> height_range,
																		  bool lidar_flip) const
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_color(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PassThrough<pcl::PointXYZ> pass;
		if (m_mapOptimiser)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr input = m_mapOptimiser->getPCLPointcloud();

			pass.setInputCloud(input);
			pass.setFilterFieldName("x");
			if (lidar_flip)
			{
				pass.setFilterLimits(-length_range[1], -length_range[0]);
			}
			else
			{
				pass.setFilterLimits(length_range[0], length_range[1]);
			}
			pass.filter(*output);

			pass.setInputCloud(output);
			pass.setFilterFieldName("y");
			pass.setFilterLimits(width_range[0], width_range[1]);
			pass.filter(*output);

			pass.setInputCloud(output);
			pass.setFilterFieldName("z");
			pass.setFilterLimits(height_range[0], height_range[1]);
			pass.filter(*output);

			pcl::copyPointCloud(*output, *output_color);

			for (auto &pt : *output_color)
			{
				pt.r = 0;
				pt.g = 255;
				pt.b = 0;
			}
			return output_color;
		}
		spdlog::warn("{} - No map optimiser exists", __func__);
		return pcl::PointCloud<pcl::PointXYZRGB>::Ptr{};
	}




	Eigen::Matrix4f LocalSlam::getCurrentPose() const
	{
		if (m_mapOptimiser)
		{
			Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
			pose.block<3, 3>(0, 0) = m_latestOdomPose.orientation.toRotationMatrix();
			pose.block<3, 1>(0, 3) = m_latestOdomPose.position;
			return pose;
		}
		spdlog::warn("No valid pose!");
		return Eigen::Matrix4f::Identity();
	}


	Eigen::Affine3f LocalSlam::get_pre_globalpose()
	{
		return m_pre_globalpose;
	}

	void LocalSlam::update_pre_globalpose(Eigen::Affine3f &transform)
	{
		m_pre_globalpose = transform;
	}




	//timestamed pose
	std::unordered_map<uint64_t, Eigen::Affine3f> LocalSlam::get_timestamped_poses() const
	{
		return m_timestamped_poses;
	}
	void LocalSlam::save_timestamped_pose_with_time(uint64_t timestamp, Eigen::Affine3f &pose)
	{
		m_timestamped_poses[timestamp] = pose;
	}
	void LocalSlam::clear_timestamped_poses()
	{
		m_timestamped_poses.clear();
	}


	//update initial transform matrix
	void LocalSlam::udpate_initial_transform(Eigen::Affine3f &transform)
	{
		m_initial_transform = transform;
	}

	void LocalSlam::reset_initial_transform_matrix()
	{
		m_initial_transform = Eigen::Affine3f::Identity();
	}

	Eigen::Affine3f LocalSlam::get_initial_transform_matrix()
	{
		return m_initial_transform;
	}

	//save bool for inisital transform
	void LocalSlam::set_transform_initialized(bool initialized)
	{
		spdlog::info("local slam initialized");
		m_transform_initialized = initialized;
	}

	bool LocalSlam::get_transform_initialized_bool() 
	{
		return m_transform_initialized;
	}
}