project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(fileio
  result_writer.cpp
)

target_link_libraries(fileio
  common
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
