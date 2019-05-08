project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(common
  container_util.cpp
  point_cloud_util.cpp
)

target_link_libraries(common
  ${OpenCV_LIBS}
)
