project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(feature_extractor
  concrete_feature_extractor_factory.cpp
  gpu/gpu_surf_feature_extractor.cpp
)

target_link_libraries(feature_extractor
  ${OpenCV_LIBS}
)
