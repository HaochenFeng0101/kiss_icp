#pragma once
#define PCL_NO_PRECOMPILE
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/conversions.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/octree/octree_search.h>
#include <pcl/point_types.h>
// #include <dashTypes/pointcloud.hpp>
// #include <boost/shared_ptr.hpp>
namespace LocalSLAM
{


  /**
     * @brief Represents a 3D pose (position and orientation).
     */
    struct Pose
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ensures correct alignment for Eigen types

        Eigen::Vector3f    position;
        Eigen::Quaternionf orientation; ///< The 3D rotational component as a unit quaternion (w, x, y, z)
		
        // Default constructor
        Pose() : position(Eigen::Vector3f::Zero()), orientation(Eigen::Quaternionf::Identity()) {}

        // Constructor with initial values
        Pose(const Eigen::Vector3f& pos, const Eigen::Quaternionf& ori)
            : position(pos), orientation(ori) {}

        // Optional: A method to convert to a 4x4 homogeneous transformation matrix
        Eigen::Matrix4f toMatrix() const {
            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = orientation.toRotationMatrix();
            T.block<3, 1>(0, 3) = position;
            return T;
        }

        // Optional: A static method to create a Pose from a 4x4 homogeneous transformation matrix
        static Pose fromMatrix(const Eigen::Matrix4f& T) {
            Pose p;
            p.position = T.block<3, 1>(0, 3);
            p.orientation = Eigen::Quaternionf(T.block<3, 3>(0, 0));
            p.orientation.normalize(); // Ensure it's a unit quaternion
            return p;
        }
    };

  struct Keyframe
  {
    std::vector<glm::vec3> points;
    glm::vec3 position;
    glm::quat rotation;
  };

  struct PoseData {
    float x{ 0.0 };
    float y{ 0.0 };
    float z{ 0.0 };

    float roll{ 0.f };
    float pitch{ 0.f };
    float yaw{ 0.f };

    std::uint64_t timestamp;
  };
}

struct dashTime {
  int sec{ 0 };
  int nsec{ 0 };
};


struct EIGEN_ALIGN16 PointXYZIRGB
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGB,
  (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
  (std::uint8_t, r, r) (std::uint8_t, g, g) (std::uint8_t, b, b)
)

typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointXYZRGBL PointTypeRGBL;

struct EIGEN_ALIGN16 PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
};                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
  (float, x, x) (float, y, y)
  (float, z, z) (float, intensity, intensity)
  (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
  (double, time, time))

typedef PointXYZIRPYT PointTypePose;

template<class PointType>
typename pcl::PointCloud<PointType>::Ptr transformPointCloud(const typename pcl::PointCloud<PointType>& cloudIn, const PointTypePose* transformIn)
{
  typename pcl::PointCloud<PointType>::Ptr cloudOut(new typename pcl::PointCloud<PointType>());
  Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
  pcl::transformPointCloud(cloudIn, *cloudOut, transCur);
  return cloudOut;
}

// using DeskewedPointCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr;