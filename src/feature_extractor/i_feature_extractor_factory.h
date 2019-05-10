
#ifndef I_FEATURE_EXTRACTOR_FACTORY_H
#define I_FEATURE_EXTRACTOR_FACTORY_H

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <feature_extractor/i_feature_extractor.h>

namespace simple_sfm {
namespace feature_extractor {

enum FeatureType : int {
  CPU_SURF,
  GPU_SURF,
  CPU_SIFT,
  GPU_SIFT,
};

class IFeatureExtractorFactory {
public:
  static cv::Ptr<IFeatureExtractor>
      createFeatureExtractor(FeatureType type);
};

}  
}
#endif // I_FEATURE_EXTRACTOR_FACTORY_H
