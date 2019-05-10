project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(visualization
  pcl_viewer.cpp
  pcl_drawer.cpp
  visualization2d_util.cpp
)

target_include_directories(visualization PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(visualization
  common
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
