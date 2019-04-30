project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(descriptor_matcher
  ConcreteDescriptorMatcherFactory.cpp
  gpu/GPUBruteForceMatcher.cpp
  gpu/GPUBruteForceMatcherWithRatioTest.cpp
)

target_link_libraries(descriptor_matcher
  ${OpenCV_LIBS}
)
