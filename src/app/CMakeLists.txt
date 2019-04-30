project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(simple_sfm 
  main.cpp
)

target_link_libraries(simple_sfm
  utility
  feature_extractor
  descriptor_matcher
  key_point_filter
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)