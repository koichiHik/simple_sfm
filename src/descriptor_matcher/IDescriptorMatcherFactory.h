
#ifndef I_DESCRIPTOR_MATCHER_FACTORY_H
#define I_DESCRIPTOR_MATCHER_FACTORY_H

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <descriptor_matcher/IDescriptorMatcher.h>

namespace simple_sfm {
namespace descriptor_matcher {

enum MatcherType : int {
  CPU_KNN,
  GPU_BF_RATIO_CHECK,
};

class IDescriptorMatcherFactory {
public:
  static cv::Ptr<IDescriptorMatcher>
      createDescriptorMatcher(MatcherType type);
};

}  
}
#endif // I_DESCRIPTOR_MATCHER_FACTORY_H
