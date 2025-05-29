#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <set>
#include <functional>
#include <sstream>
#include <map>

#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "localSlam.h"
#include "overgrowth_detection.h"

LocalSLAM::LocalSlam _localSlam;
std::unique_ptr<LocalSLAM::OvergrowthDetector> _overgrowth_ptr;

uint64_t last_lidar_timestamp = 0;
uint64_t updated_lidar_timestamp = 0;
uint64_t last_pose_timestamp = 0;
uint64_t _preTimestamp{0};

float slam_time = 20.0f;
bool isnew_slam = true;

const float _angle_threshold = 30.0f * M_PI / 180.0f;
const float _detection_box_length = 1.0f;
const float _detection_box_width = 0.7f;
const float _detection_box_height = 1.0f;
float _search_height = 1.9f; // seach height - lidar height1.9
float _lidar_height = 0.3f;
int _points_threshold = 10; // 400

Eigen::Affine3f _current_world_transform = Eigen::Affine3f::Identity();
Eigen::Affine3f current_global_transform = Eigen::Affine3f::Identity();
Eigen::Affine3f _pre_global_pose = Eigen::Affine3f::Identity();

pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr pose_list(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ros_pose(new pcl::PointCloud<pcl::PointXYZ>);

Eigen::Affine3f _lidar_flip_mat = Eigen::Affine3f::Identity();
bool lidar_flip = false;
bool lidar_extrinsics = true;

Eigen::Affine3f lidar_extrinsic = Eigen::Affine3f::Identity();
float extrinsic_x = 0.0f;
float extrinsic_y = 0.0f;
float extrinsic_z = 0.0f;
float extrinsic_roll = 0.0f;
float extrinsic_pitch = 0.0f;
float extrinsic_yaw = 45.0f;
// Convert degrees to radians
float roll_rad = extrinsic_roll * M_PI / 180.0f;
float pitch_rad = extrinsic_pitch * M_PI / 180.0f;
float yaw_rad = extrinsic_yaw * M_PI / 180.0f;

bool initialize()
{
    _overgrowth_ptr = std::make_unique<LocalSLAM::OvergrowthDetector>(
        _detection_box_length,
        _detection_box_width,
        _detection_box_height,
        _points_threshold,
        _search_height,
        _lidar_height);

    return true;
}

void poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    _current_world_transform.translation() = Eigen::Vector3f(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    Eigen::Quaternionf q(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);
    _current_world_transform.linear() = q.toRotationMatrix();

    pcl::PointXYZ point_ros;
    point_ros.x = _current_world_transform.translation().x();
    point_ros.y = _current_world_transform.translation().y();
    point_ros.z = _current_world_transform.translation().z();
    ros_pose->push_back(point_ros);
    // pcl::io::savePCDFileBinaryCompressed("/home/haochen/dconstruct/dash_robot/build/dash_code/pointcloud_utils/localSlam/test/map/pose_ros.pcd", *ros_pose);
}

void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    // get current cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    last_lidar_timestamp = msg->header.stamp.toNSec();
    pcl::copyPointCloud(*cloud, *latest_cloud);
}

void Tick()
{
    if (last_lidar_timestamp != updated_lidar_timestamp)
    {
        // submit latest cloud
        _localSlam.submit(latest_cloud);
        updated_lidar_timestamp = last_lidar_timestamp;
        const float timeDiffSec = static_cast<float>(updated_lidar_timestamp - _preTimestamp) / 1e9;
        // spdlog::info("time diff: {}, updated_lidar_timestamp: {}, preTimestamp: {}", timeDiffSec, updated_lidar_timestamp, _preTimestamp);

        // check if new slam, then grab the current local to world transform

        if (isnew_slam)
        {
            current_global_transform = _current_world_transform;
            isnew_slam = false;
            // spdlog::info("recorded world transform!!!");
        }
        

        Eigen::Affine3f current_local_pose = Eigen::Affine3f::Identity();

        current_local_pose.matrix() = _localSlam.getCurrentPose();
        // check where need to flip pose
        if (lidar_flip)
        {
            _lidar_flip_mat.matrix() << -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;
            // current_world_transform = _current_world_transform ;
            // current_local_pose = current_global_transform * _lidar_flip_mat * current_local_pose;
            current_local_pose = _lidar_flip_mat * current_local_pose;
        }
        else
        {
            // current_local_pose = current_global_transform * current_local_pose;
            current_local_pose = current_local_pose;
        }


        if (lidar_extrinsics)
        {
            lidar_extrinsic.translation() << extrinsic_x, extrinsic_y, extrinsic_z;

            Eigen::Matrix3f rotation;
            rotation =
                Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(pitch_rad, Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(roll_rad, Eigen::Vector3f::UnitX());
        
            lidar_extrinsic.linear() = rotation;
            //times lidar extrinsic
            current_local_pose = lidar_extrinsic * current_local_pose;
        }
    
        

        // save local slam global transformed pose and time stamp
        if (_localSlam.get_timestamped_poses().empty())
        {
            _localSlam.update_pre_globalpose(current_local_pose);
            uint64_t current_timestamp = updated_lidar_timestamp;
            _localSlam.save_timestamped_pose_with_time(current_timestamp, current_local_pose);
        }
        else
        {
            _pre_global_pose = _localSlam.get_pre_globalpose();
            const auto pose_diff = _overgrowth_ptr->getDistanceYawDifference(_pre_global_pose, current_local_pose);
            if (pose_diff.first > _detection_box_length || abs(pose_diff.second) > _angle_threshold)
            {
                _localSlam.update_pre_globalpose(current_local_pose);
                // Store pose with timestamp
                uint64_t current_timestamp = updated_lidar_timestamp;
                _localSlam.save_timestamped_pose_with_time(current_timestamp, current_local_pose);
            }
        }
        
        // for debug
        pcl::PointXYZ point;
        Eigen::Affine3f global_pose = current_global_transform * current_local_pose;
        point.x = global_pose.translation().x();
        point.y = global_pose.translation().y();
        point.z = global_pose.translation().z();
        pose_list->push_back(point);

        if (timeDiffSec > slam_time)
        {
            if (_preTimestamp == 0)
            {
                _preTimestamp = updated_lidar_timestamp;
                spdlog::warn("No previous timestamp, skipping...");
                return;
            }

            spdlog::info("new localslam session");

            // for debug
            std::string filename_pose = "/home/haochen/dconstruct/kiss_icp/result_folder/pose_data.pcd";
            pcl::io::savePCDFileBinaryCompressed(filename_pose, *pose_list);

            // Get current map and transform to world frame
            pcl::PointCloud<pcl::PointXYZ>::Ptr local_map = _localSlam.getLocalMap();
            // pcl::transformPointCloud(*local_map, *local_map, current_global_transform);
            // pcl::io::savePCDFileBinaryCompressed("/home/haochen/dconstruct/dash_robot/build/dash_code/pointcloud_utils/localSlam/test/map/localmap" + std::to_string(updated_lidar_timestamp) + ".pcd", *local_map);

            // update local map
            _overgrowth_ptr->updateMap(local_map);

            // get box center and added to tree data
            for (const auto &[timestamp, pose] : _localSlam.get_timestamped_poses())
            {
                LocalSLAM::OverGrowthTreeData data;
                data.pose = pose;
                data.timestamp = timestamp;
                // spdlog::info("cur timestamp: {}", timestamp);
                data.is_overgrown = false; // Will be set by overgrowthDetection
                _overgrowth_ptr->addOvergrownTreeData(data);
            }

            // spdlog::info("overgrown data size {}", _overgrowth_ptr->getTreeDataList().size());

            bool has_overgrowth = _overgrowth_ptr->overgrowthDetection();

            if (has_overgrowth)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr overgrown_points(new pcl::PointCloud<pcl::PointXYZ>);

                overgrown_points = _overgrowth_ptr->getOvergrownPoints();
                pcl::transformPointCloud(*overgrown_points, *overgrown_points, current_global_transform);
                spdlog::warn("Overgrown detected with size {}", overgrown_points->size());
                pcl::io::savePCDFileBinaryCompressed("/home/haochen/dconstruct/kiss_icp/result_folder/overgrown_points" + std::to_string(updated_lidar_timestamp) + ".pcd", *overgrown_points);
                // overgrown_start_timestamp = _overgrowth_ptr->getOvergrownStartTimestamp();
                // overgrown_end_timestamp = _overgrowth_ptr->getOvergrownEndTimestamp();
                
            }

            // reset slam
            isnew_slam = true;
            _preTimestamp = updated_lidar_timestamp;
            // spdlog::info("previous timestamp updated to {}", _preTimestamp);
            _localSlam.resetSlam();
            _overgrowth_ptr->clearovergrownData();
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "localslam_node");

    // Create a NodeHandle
    ros::NodeHandle nh;

    if (!initialize())
    {
        return -1;
    }

    ros::Subscriber cloud_sub = nh.subscribe("/dc/lidar/main/xyzirt/sensor_frame", 10, cloudCallback);
    ros::Subscriber pose_sub = nh.subscribe("/dc/loc/pose", 10, poseCallback);
    spdlog::info("Starting ROS node, please play rosbag file...");

    while (ros::ok())
    {
        ros::spinOnce();
        Tick();
    }

    return 0;
}