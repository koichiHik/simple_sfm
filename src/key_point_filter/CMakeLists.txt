project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(key_point_filter
  ConcreteKeyPointFilterFactory.cpp
  cpu/FMatKeyPointFilter.cpp
  cpu/HomographyKeyPointFilter.cpp
)

target_link_libraries(key_point_filter
  ${OpenCV_LIBS}
)
