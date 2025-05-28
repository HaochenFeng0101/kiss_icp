# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
find_package(OpenMP REQUIRED)
add_library(core STATIC)
add_library(kiss_icp::core ALIAS core)
target_sources(core PRIVATE Registration.cpp Deskew.cpp Preprocessing.cpp Threshold.cpp VoxelHashAndAlign.cpp)
target_link_libraries(core PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb Sophus::Sophus OpenMP::OpenMP_CXX)
set_global_target_properties(core)

IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    # target_compile_options(core PRIVATE -march=native -ffast-math) # march native broke on jetson lol
    target_compile_options(core PRIVATE -march=armv8.2-a+crypto+fp16+rcpc+dotprod -ffast-math)
    # target_compile_options(core PRIVATE -ffast-math)
ELSE()
    target_compile_options(core PRIVATE -ffast-math)
ENDIF()
