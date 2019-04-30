project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(feature_extractor
  ConcreteFeatureExtractorFactory.cpp
  gpu/GPUSurfFeatureExtractor.cpp
)

target_link_libraries(feature_extractor
  ${OpenCV_LIBS}
)
