project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(vis3d
  PCLViewer.cpp
  PCLDrawer.cpp
)

target_include_directories(vis3d PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(vis3d
  common
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
