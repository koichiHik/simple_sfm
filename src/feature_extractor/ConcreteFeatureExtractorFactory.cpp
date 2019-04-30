
// System
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <feature_extractor/IFeatureExtractorFactory.h>
#include <feature_extractor/ConcreteFeatureExtractor.h>

namespace simple_sfm {
namespace feature_extractor {

cv::Ptr<IFeatureExtractor>
IFeatureExtractorFactory::
  createFeatureExtractor(FeatureType type) {

    cv::Ptr<IFeatureExtractor> ptr;
    switch (type) {
    case FeatureType::GPU_SURF:
      ptr = new GPUSurfFeatureExtractor();
      break;
    default:
      std::cerr << "Specified feature type is not supported...." << std::endl;
      std::exit(0);
    }
  
  return ptr;
}

}  
}