project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_executable(simple_sfm
  fmat_point3d_gen_runner.cpp
  match_filtering_runner.cpp
  pnp_point3d_gen_runner.cpp
  point2d_matching_runner.cpp
  sfm_runner.cpp
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
  visualization
  common
  fileio
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)
