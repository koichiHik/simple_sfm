
// System
#include <iostream>

// OpenCV


// Original
#include <feature/IFeatureCalculatorFactory.h>
#include <feature/ConcreteFeatureCalculator.h>

namespace simple_sfm {
namespace feature {

cv::Ptr<IFeatureCalculator>
IFeatureCalculatorFactory::
  createFeatureCalculator(FeatureType type) {

    cv::Ptr<IFeatureCalculator> ptr;
    switch (type) {
    case FeatureType::GPU_SURF:
      ptr = new GPUSurfFeatureCalculator();
      break;
    default:
      std::cerr << "Specified feature type is not supported...." << std::endl;
      std::exit(0);
    }
  
  return ptr;
}

}  
}