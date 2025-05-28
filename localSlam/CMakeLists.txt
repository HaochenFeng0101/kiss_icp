cmake_minimum_required(VERSION 3.15)
include_guard(GLOBAL)
project(localSlam)

find_package(spdlog CONFIG REQUIRED)
find_package(GTSAM CONFIG REQUIRED QUIET)
find_package(PCL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glm CONFIG REQUIRED)

add_subdirectory(../kiss_icp kiss_icp)
add_library(localSlam
    src/localSlam.cpp
    src/lidarOdom.cpp
    src/mapOptimiser.cpp
)
# add_library(dash::localSlam ALIAS localSlam)

target_link_libraries(localSlam 
PUBLIC
    ${PCL_LIBRARIES}
    kiss_icp::pipeline
    kiss_icp::core
    kiss_icp::metrics
    glm::glm
    ${catkin_LIBRARIES} #ros

PRIVATE
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    gtsam
    Eigen3::Eigen
)

target_include_directories(localSlam 
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    
    ${GTSAM_INCLUDE_DIR}
PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)

if (UNIX)
  add_subdirectory(test)
endif()
