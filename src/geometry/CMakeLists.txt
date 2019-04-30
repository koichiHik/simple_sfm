project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(geometry
  Geometry.cpp
)

target_link_libraries(geometry
  common
  ${OpenCV_LIBS}
)
