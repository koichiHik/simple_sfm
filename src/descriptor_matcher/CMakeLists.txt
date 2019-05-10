project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(descriptor_matcher
  concrete_descriptor_matcher_factory.cpp
  gpu/gpu_brute_force_matcher.cpp
  gpu/gpu_brute_force_matcher_with_ratio_test.cpp
)

target_link_libraries(descriptor_matcher
  ${OpenCV_LIBS}
)
