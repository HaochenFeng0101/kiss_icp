#pragma once

#include <array>
#include <atomic>
#include <string>
// #include <flodom/common_types.hpp>
#include "KissICP.hpp"

namespace LocalSLAM
{
	struct MapOptimiserConfig
	{
		float positionDelta = 0.5f;
		float rollThreshold = 0.1f; 
		float pitchThreshold = 0.1f; 
		float yawThreshold = 0.1f;
    float voxelSize = 0.2f; // Voxel Size used to downsample pointcloud for caching
	};

	struct LidarOdomConfig{
    // Whether cameras are on
    bool preview_image = true;
    // Number of cameras
    int num_cameras = 4;
    // Sensor Timeout in milliseconds
    int sensor_timeout = 200;
    // Drop Threshold. Number of consecutive timeouts of a sensor before it is dropped.
    int drop_sensor_threshold = 50; // 50 implies approx 5s of downtime.
    // Max timediff in syncing in nanoseconds
    int max_timediff = 100000000;
    // Number of lidar data to ignore between enqueues.
    bool lidar_dynamic_throttle = false; // Throttle based on queue size
    // IMU Preintegration Parameters
    bool use_imu = false;
    float lidar_pitch = 30.0f * M_PI / 180.0f; // +ve = pitch backward(+ve x point to sky). (Inverted if you treat rotation as about y axis)
    // IMU Noise Covariances
    double imu_accelerometer_sd     = 3.9939570888238808e-3f;
    double imu_gyroscope_sd         = 1.5636343949698187e-3f;
    double imu_accelerometer_bias   = 6.4356659353532566e-5f;
    double imu_gyroscope_bias       = 3.5640318696367613e-5f;
    double imu_gravity             = -9.81;

    // Run deskew (Results seems worse with current deskew)
    bool run_deskew = false;

    // Lidar Odometry Configs
    std::string lidar_odom_mode = "kiss_icp"; // "flodom" or "kiss_icp"

    // Kiss ICP setting
    kiss_icp::pipeline::KISSConfig kiss_icp_config{ 0.4, // Voxel Size
                                                    15.0, // Max Range Of Lidar
                                                    1.0, // Min Range of Lidar
                                                    10, // Max Points per Voxel
                                                    1500, // Max Points used for ICP
                                                    0.1, // Minimum Motion Threshold
                                                    2.0, // Initial Threshold
                                                    false}; // Deskew
    
    // IMU Convergence Criterion. Needs to be strict, else deskew will screw up.
    int max_icp_points_registered = 8000;
    float imu_converge_dist_threshold     = 0.2f;                  // m
    float imu_converge_rotation_threshold = 15.0f * M_PI / 180.0f; // rad
    int min_converge_iterations_threshold = 100;
    
    // Keypoint Criterion
    float keypoint_dist_threshold   = 0.5f;    // m
    float keypoint_turn_threshold   = 20.0f * M_PI / 180.0f;  // rad
    int64_t keypoint_time_threshold = 1000000; // µs

    // Historical Key Pose Buffer
    size_t key_pose_buffer_size = 5000;

    // gRPC settings
    std::string stream_topic = "/local_slam/pointcloud";

    // Downsampling Preview
    float preview_crop_box = 30.0f;
    float preview_voxel_ds_size = 0.1f;
    int points_streamed = 5000;
    float preview_image_factor = 0.5f;
    float preview_image_jpg_quality = 50;

    // Cam only preview frequency
    float cam_preview_frequency = 2.0f; // In Hz.
	};


  struct LocalSlamConfig {
    LidarOdomConfig lidarOdomConfig;
    MapOptimiserConfig mapOptimConfig;
  };


}