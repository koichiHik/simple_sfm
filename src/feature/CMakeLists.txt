project(simple_sfm)
cmake_minimum_required(VERSION 2.8)

add_library(feature
  ConcreteFeatureCalculatorFactory.cpp
  gpu/GPUSurfFeatureCalculator.cpp
)

target_link_libraries(feature
  ${OpenCV_LIBS}
)
