project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_executable(simple_sfm 
  main.cpp
)

target_include_directories(simple_sfm PRIVATE
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(simple_sfm
  utility
  feature_extractor
  descriptor_matcher
  key_point_filter
  geometry
  vis3d
  common
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)