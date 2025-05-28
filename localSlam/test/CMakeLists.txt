cmake_minimum_required(VERSION 3.15)
# include_guard(GLOBAL)
project(kiss_icp_rosbag_test)

# # To find ROS dependencies
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../../../../dash_code/CMakeModules")

# Find required packages
find_package(PCL CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED) 
find_package(catkin REQUIRED COMPONENTS
    roscpp
    roscpp_serialization
    sensor_msgs
    geometry_msgs
    rosbag
    tf2
    pcl_conversions
)

# Add localSlam library
add_subdirectory(../. localSlam)

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    
)

# Add your executable
add_executable(rosbag_test main.cpp)

target_compile_features(rosbag_test PUBLIC cxx_std_17)
target_link_libraries(rosbag_test
PRIVATE
    localSlam
    spdlog::spdlog
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
)