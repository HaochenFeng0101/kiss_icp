#pragma once

#include <mutex>
#include <pcl/io/pcd_io.h>
#include <set>

#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/common/io.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/transformation_estimation.h>
#include <string>

namespace LocalSLAM
{
    struct OverGrowthTreeData
    {
        Eigen::Affine3f pose;
        bool is_overgrown;
        uint64_t timestamp;
    };

    class OvergrowthDetector
    {
    public:
        OvergrowthDetector(float detection_box_length,
                           float detection_box_width,
                           float detection_box_height,
                           int points_threshold,
                           float search_height,
                           float lidar_height)
            : m_points_threshold(points_threshold),
              m_search_height(search_height),
              m_lidar_height(lidar_height)
        {
            m_overgrown_points.reset(new pcl::PointCloud<pcl::PointXYZ>);
            m_overgrown_box_search_x = detection_box_length;
            m_overgrown_box_search_y = detection_box_width;
            m_overgrown_box_search_z = detection_box_height;
            spdlog::info("Overgrowth Detector Constructed");
        }

        void updateMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &map)
        // update current map
        {
            m_kd_tree.setInputCloud(map);
        }

        bool overgrowthDetection()
        {
            if (m_tree_data_list.empty())
            {
                spdlog::warn("No poses provided for overgrowth detection");
                return false;
            }

            // For each map, match the first box center to the current map, then check box
            int overgrown_count = 0;
            const pcl::PointCloud<pcl::PointXYZ>::ConstPtr input_cloud = m_kd_tree.getInputCloud();

            for (auto &tree_data : m_tree_data_list)
            {
                const auto &pose_matrix = tree_data.pose;
                Eigen::Vector3f tree_pos = pose_matrix.translation();
                Eigen::Matrix3f R = pose_matrix.rotation();
                Eigen::Quaternionf quaternion(R);

                float tree_x = tree_pos.x();
                float tree_y = tree_pos.y();
                float tree_z = tree_pos.z() - m_lidar_height + m_search_height;
                Eigen::Vector3f tree_pos_3f(tree_x, tree_y, tree_z);
                
                // filter points in local coordinates (the box is axis-aligned)
                pcl::PointCloud<pcl::PointXYZ>::Ptr overgrown_points(new pcl::PointCloud<pcl::PointXYZ>);

                for (size_t j = 0; j < input_cloud->size(); ++j)
                {
                    const auto &pt = (*input_cloud)[j];
                    Eigen::Vector3f pt_local;
                    pt_local = R.transpose() * (Eigen::Vector3f(pt.x, pt.y, pt.z) - tree_pos_3f);
                    
                    if (pt_local.x() >= -m_overgrown_box_search_x && pt_local.x() <= m_overgrown_box_search_x &&
                        pt_local.y() >= -m_overgrown_box_search_y && pt_local.y() <= m_overgrown_box_search_y &&
                        pt_local.z() >= 0.0f && pt_local.z() < 3.0f)
                    {
                        // Index j corresponds to same point in input_cloud
                        overgrown_points->push_back((*input_cloud)[j]);
                    }
                }

                // Check if number of points exceeds threshold
                if (int(overgrown_points->size()) > m_points_threshold)
                {

                    tree_data.is_overgrown = true;
                    if (overgrown_count == 0)
                    {
                        m_overgrown_start_timestamp = tree_data.timestamp;
                    }
                    m_overgrown_end_timestamp = tree_data.timestamp;
                    overgrown_count++;
                    *m_overgrown_points += *overgrown_points;

                    // save points for debug
                    // std::string filename_overgrown_points = "/home/haochen/dconstruct/dash_robot/dash_code/pointcloud_utils/localSlam/build/map/points_" + std::to_string(m_tree_data_list[0].timestamp) + ".pcd";
                    // pcl::io::savePCDFileBinaryCompressed(filename_overgrown_points, *overgrown_points);
                    // spdlog::warn("overgrown detected and size {} ", overgrown_points->size());
                }
                else
                {
                    tree_data.is_overgrown = false;
                }
            }

            return overgrown_count > 0;
        }

        std::pair<float, float> getDistanceYawDifference(const Eigen::Affine3f &pre_pose, const Eigen::Affine3f &curr_pose) const
        // check distance and angle between two pose, can know if it is turning a lot
        {
            const Eigen::Vector3f translation1 = pre_pose.translation();
            const Eigen::Vector3f translation2 = curr_pose.translation();

            // Calculate Euclidean distance
            const float euclidean_diff = (translation1 - translation2).norm();

            // Extract rotation matrices (top-left 3x3 block of the 4x4 matrix)
            const Eigen::Matrix3f rotation1 = pre_pose.rotation();
            const Eigen::Matrix3f rotation2 = curr_pose.rotation();

            // Calculate yaw for both poses
            const float yaw1 = get_yaw(rotation1);
            const float yaw2 = get_yaw(rotation2);

            // Calculate yaw difference, normalized to the range [-pi, pi]
            float yaw_diff = yaw2 - yaw1;
            yaw_diff = std::atan2(std::sin(yaw_diff), std::cos(yaw_diff)); // Normalize the angle

            return std::pair<float, float>(euclidean_diff, yaw_diff);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr getOvergrownPoints() const
        {
            return m_overgrown_points;
        }

        uint64_t getOvergrownStartTimestamp() const
        {
            return m_overgrown_start_timestamp;
        }

        uint64_t getOvergrownEndTimestamp() const
        {
            return m_overgrown_end_timestamp;
        }

        std::vector<OverGrowthTreeData> getTreeDataList() const
        {
            return m_tree_data_list;
        }
        void addOvergrownTreeData(const OverGrowthTreeData &data)
        {
            m_tree_data_list.push_back(data);
        }

        void clearovergrownData()
        {
            m_tree_data_list.clear();
            m_overgrown_points->clear();
            m_overgrown_start_timestamp = 0.0;
            m_overgrown_end_timestamp = 0.0;
        }

    protected:
        float get_yaw(const Eigen::Matrix3f &rotation) const
        {
            return std::atan2(rotation(1, 0), rotation(0, 0));
        }

    private:
        pcl::search::KdTree<pcl::PointXYZ> m_kd_tree;
        pcl::PointCloud<pcl::PointXYZ>::Ptr m_overgrown_points;
        std::vector<OverGrowthTreeData> m_tree_data_list;

        float m_overgrown_box_search_x = 2.5f;
        float m_overgrown_box_search_y = 2.5f;
        float m_overgrown_box_search_z = 1.5f;

        int m_points_threshold = 200;
        float m_search_height = 1.9f;
        float m_lidar_height = 0.3f;

        uint64_t m_overgrown_start_timestamp = 0;
        uint64_t m_overgrown_end_timestamp = 0;
    };
}