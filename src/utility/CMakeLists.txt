project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

include_directories(${Boost_INCLUDE_DIRS})

add_library(utility
  fileutil.cpp
)

target_link_libraries(utility
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)
