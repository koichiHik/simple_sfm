
#ifndef I_FEATURE_CALCULATOR_FACTORY_H
#define I_FEATURE_CALCULATOR_FACTORY_H

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <feature/IFeatureCalculator.h>

namespace simple_sfm {
namespace feature {

enum FeatureType : int {
  CPU_SURF,
  GPU_SURF,
  CPU_SIFT,
  GPU_SIFT,
};

class IFeatureCalculatorFactory {
public:
  static cv::Ptr<IFeatureCalculator>
      createFeatureCalculator(FeatureType type);
};

}  
}
#endif // I_FEATURE_CALCULATOR_FACTORY_H
