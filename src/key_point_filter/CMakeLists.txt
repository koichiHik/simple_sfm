project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(key_point_filter
  concrete_key_point_filter_factory.cpp
  cpu/fmat_key_point_filter.cpp
  cpu/homography_key_point_filter.cpp
)

target_link_libraries(key_point_filter
  ${OpenCV_LIBS}
)
