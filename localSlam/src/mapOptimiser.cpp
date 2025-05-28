#include "mapOptimiser.h"

#include "glm/gtx/norm.hpp"

#include <gtsam/linear/NoiseModel.h>
#include <spdlog/spdlog.h>
#include <pcl/common/eigen.h>
#include <pcl/common/impl/eigen.hpp>
#include <pcl/filters/voxel_grid.h>

namespace
{
	glm::vec3 getEuler(glm::mat4 const& mat)
	{
		glm::vec3 result;

		result.x = std::atan2(-mat[2][3], mat[2][2]);
		result.y = std::asin(mat[2][1]);
		result.z = std::atan2(mat[3][1], mat[1][1]);

		return result;
	}

	gtsam::Pose3 glmToGtsamPost(glm::vec3 const& position, glm::quat const& rotation)
	{
		auto const euler = glm::eulerAngles(rotation);

		return gtsam::Pose3(
			gtsam::Rot3::RzRyRx(euler.x, euler.y, euler.z),
			gtsam::Point3(position.x, position.y, position.z)
		);
	}
}

namespace LocalSLAM
{
	MapOptimiser::MapOptimiser(const MapOptimiserConfig& configIn)
	:m_config(configIn)
	{
		m_maxKeyframeDistSquare = m_config.positionDelta * m_config.positionDelta;
	}

	bool MapOptimiser::updatePoseGraphWithKeyframes(
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcPtr,
		Eigen::Vector3f              position,
		Eigen::Quaternionf           rotation,
		uint64_t                     timestamp
	)
	{
		Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
		tf.block<3, 3>(0, 0) = rotation.toRotationMatrix();
		tf.block<3, 1>(0, 3) = position;
		Eigen::Affine3f tfAffine(tf);
		Eigen::Affine3f correctedWorldTScan = mWorldTOdomWorld * tfAffine;
		pcl::getTranslationAndEulerAngles(correctedWorldTScan, m_transformToBeMapped[3], m_transformToBeMapped[4], m_transformToBeMapped[5], m_transformToBeMapped[0], m_transformToBeMapped[1], m_transformToBeMapped[2]);
		m_currentTimestamp       = timestamp;

		if (!checkShouldCache())
		{
			return false;
		}
		m_mapGraph.resize(0);
		m_mapInitGraphValues.clear();

		addOdomFactor();

		try
		{
			// update iSAM
			m_mapOptimiser.update(m_mapGraph, m_mapInitGraphValues);
			m_mapOptimiser.update();
		}
		catch (const gtsam::IndeterminantLinearSystemException& ex)
		{
			spdlog::error("{} - Internal Linear System error {}", __func__, ex.what());
			// size_t badKey = static_cast<size_t>(ex.nearbyVariable());
			// removeFactorGraphKey(ex.nearbyVariable());
			std::cout << "****************************************************" << "\n";
			m_mapGraph.print("GTSAM Graph:\n");
			return false;
		}
		catch (const gtsam::ValuesKeyAlreadyExists& ex)
		{
			spdlog::info("{} - Internal Key error {}", __func__, ex.what());
			// removeFactorGraphKey(ex.key());
			return false;
		}
		catch (const std::exception& ex)
		{
			spdlog::error("{} - Internal Error {}", __func__, ex.what());
			return false;
		}
		//save key poses
		m_mapCurrGraphValues = m_mapOptimiser.calculateEstimate();

		// Downsample pointcloud for caching
		pcl::VoxelGrid<pcl::PointXYZ> voxGrid;
		voxGrid.setLeafSize(m_config.voxelSize, m_config.voxelSize, m_config.voxelSize);
		voxGrid.setInputCloud(pcPtr);
		pcl::PointCloud < pcl::PointXYZ >::Ptr dsPCPtr{ new pcl::PointCloud<pcl::PointXYZ>() };
		voxGrid.filter(*dsPCPtr);

		{
			std::unique_lock lock(m_keyPoseMtx);
			updateAllPoses();
			m_keyframePointclouds.push_back(dsPCPtr);
		}
		updateWorldTOdomTransform(tfAffine);

		return true;
	}


	void MapOptimiser::updateWorldTOdomTransform(const Eigen::Affine3f& odomWorldTscan)
	{
		const Eigen::Affine3f worldTscan = getKeyframeAffine3f(getNumKeyframes() - 1);
		mWorldTOdomWorld = worldTscan * odomWorldTscan.inverse();
	}

	size_t MapOptimiser::getNumKeyframes() const
	{
		std::shared_lock lock(m_keyPoseMtx);
		return m_keyframePointclouds.size();
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr MapOptimiser::getPCLPointcloud() const
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcPtrOut{new pcl::PointCloud<pcl::PointXYZ>()};
		pcl::PointCloud<pcl::PointXYZ> tmpPC;

		size_t numKeyframes = getNumKeyframes();
		for(int frameID = 0 ; frameID < numKeyframes; frameID++){
			auto pcPtr =getPointCloudPtrAtFrame(frameID);
			Eigen::Affine3f tf = getKeyframeAffine3f(frameID);
			pcl::transformPointCloud(*pcPtr, tmpPC, tf);
			*pcPtrOut += tmpPC;
		}
		return pcPtrOut;
	}

	Eigen::Affine3f MapOptimiser::getKeyframeAffine3f(size_t keyframeID) const
	{
		std::shared_lock lock(m_keyPoseMtx);
		return poseDataToAffine3f(m_keyPoseWithTimestamp[keyframeID]);
	}

	std::vector<Eigen::Vector3f> MapOptimiser::getPointcloudEigenVector3f() const
	{
		auto pcPtr = getPCLPointcloud();
		std::vector<Eigen::Vector3f> ptListOut;
		ptListOut.resize(pcPtr->size());

		#pragma omp parallel for
		for(int i = 0 ; i < pcPtr->size(); i++){
			Eigen::Vector3f& target = ptListOut[i];
			const auto& pt = pcPtr->at(i);
			target.x() = pt.x;
			target.y() = pt.y;
			target.z() = pt.z;
		}
		return ptListOut;
	}

	std::vector<Eigen::Vector3d> MapOptimiser::getPointcloudEigenVector3d() const
	{
		auto pcPtr = getPCLPointcloud();
		std::vector<Eigen::Vector3d> ptListOut;
		ptListOut.resize(pcPtr->size());

		#pragma omp parallel for
		for(int i = 0 ; i < pcPtr->size(); i++){
			Eigen::Vector3d& target = ptListOut[i];
			const auto& pt = pcPtr->at(i);
			target.x() = static_cast<double>(pt.x);
			target.y() = static_cast<double>(pt.y);
			target.z() = static_cast<double>(pt.z);
		}
		return ptListOut;
	}

	std::vector<PoseData> const& MapOptimiser::getPoses() const
	{
		return m_keyPoseWithTimestamp;
	}

	std::vector<PoseData> MapOptimiser::getKeyframePosesCpy() const
	{
		std::shared_lock lock(m_keyPoseMtx);
		return m_keyPoseWithTimestamp;
	}

	void MapOptimiser::updateAllPoses()
	{
		m_keyPoseWithTimestamp.push_back(
			PoseData{
				static_cast<float>(m_transformToBeMapped[3]),
				static_cast<float>(m_transformToBeMapped[4]),
				static_cast<float>(m_transformToBeMapped[5]),
				static_cast<float>(m_transformToBeMapped[0]),
				static_cast<float>(m_transformToBeMapped[1]),
				static_cast<float>(m_transformToBeMapped[2]),
				m_currentTimestamp
			}
		);


		for (size_t frameID = 0; frameID < m_mapCurrGraphValues.size(); frameID++)
		{
			auto&       toUpdate = m_keyPoseWithTimestamp[frameID];
			auto const& source   = m_mapCurrGraphValues.at<gtsam::Pose3>(frameID);

			toUpdate.x = static_cast<float>(source.translation().x());
			toUpdate.y = static_cast<float>(source.translation().y());
			toUpdate.z = static_cast<float>(source.translation().z());

			toUpdate.roll  = static_cast<float>(source.rotation().roll());
			toUpdate.pitch = static_cast<float>(source.rotation().pitch());
			toUpdate.yaw   = static_cast<float>(source.rotation().yaw());
		}
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr const MapOptimiser::getPointCloudPtrAtFrame(int frameID) const
	{
		std::shared_lock lock(m_keyPoseMtx);
		if(frameID >= m_keyPoseWithTimestamp.size()){
			return nullptr;
		}

		return m_keyframePointclouds[frameID];
	}


	bool MapOptimiser::checkShouldCache() const
	{
		if (m_keyPoseWithTimestamp.size() < 20) // Keep the first 20 frames
		{
			return true;
		}

		Eigen::Affine3f transStart = poseDataToAffine3f(m_keyPoseWithTimestamp.back());
		Eigen::Affine3f transFinal = pcl::getTransformation(
			m_transformToBeMapped[3],
			m_transformToBeMapped[4],
			m_transformToBeMapped[5],
			m_transformToBeMapped[0],
			m_transformToBeMapped[1],
			m_transformToBeMapped[2]
		);
		Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
		float           x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

		return
			std::abs(roll) >= m_config.pitchThreshold ||
			std::abs(pitch) >= m_config.yawThreshold ||
			std::abs(yaw) >= m_config.rollThreshold ||
			glm::length2(glm::vec3(x, y, z)) >= m_maxKeyframeDistSquare;
	}

	void MapOptimiser::addOdomFactor()
	{
		size_t const newFactorID = m_keyPoseWithTimestamp.size();

		if (m_keyPoseWithTimestamp.empty())
		{
			gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
				(gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()
			); // rad*rad, meter*meter

			auto const gtsamPose = trans2gtsamPose(m_transformToBeMapped);

			m_mapGraph.add(gtsam::PriorFactor<gtsam::Pose3>(newFactorID, gtsamPose, priorNoise));
			m_mapInitGraphValues.insert(newFactorID, gtsamPose);
		}
		else
		{
			gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(
				(gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished()
			);
			gtsam::Pose3 const poseFrom = poseDataToGtsamPose3(m_keyPoseWithTimestamp.back());
			gtsam::Pose3 const poseTo   = trans2gtsamPose(m_transformToBeMapped);
			// newFactorID - 1 is safe because newFactorID >= 1
			m_mapGraph.add(
				gtsam::BetweenFactor<gtsam::Pose3>(newFactorID - 1, newFactorID, poseFrom.between(poseTo), odometryNoise)
			);
			m_mapInitGraphValues.insert(newFactorID, poseTo);
		}
	}

	Eigen::Affine3f MapOptimiser::poseDataToAffine3f(const PoseData& thisPoint)
	{
		return pcl::getTransformation(
			thisPoint.x,
			thisPoint.y,
			thisPoint.z,
			thisPoint.roll,
			thisPoint.pitch,
			thisPoint.yaw
		);
	}

	gtsam::Pose3 MapOptimiser::poseDataToGtsamPose3(const PoseData& thisPoint)
	{
		return gtsam::Pose3(
			gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
			gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z))
		);
	}

	gtsam::Pose3 MapOptimiser::trans2gtsamPose(float transformIn[])
	{
		return gtsam::Pose3(
			gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
			gtsam::Point3(transformIn[3], transformIn[4], transformIn[5])
		);
	}

	gtsam::Pose3 MapOptimiser::eigenAffine2gtsamPose(const Eigen::Affine3f& affine)
	{
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(affine, x, y, z, roll, pitch, yaw);
		return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
	}

}
