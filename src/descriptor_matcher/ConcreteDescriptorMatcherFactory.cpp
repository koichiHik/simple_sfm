
// System
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <descriptor_matcher/IDescriptorMatcherFactory.h>
#include <descriptor_matcher/ConcreteDescriptorMatcher.h>

namespace simple_sfm {
namespace descriptor_matcher {

cv::Ptr<IDescriptorMatcher>
IDescriptorMatcherFactory::
  createDescriptorMatcher(MatcherType type) {

    cv::Ptr<IDescriptorMatcher> ptr;
    switch (type) {
    case MatcherType::GPU_BF_RATIO_CHECK:
      ptr = new GPUBruteForceMatcherWithRatioCheck();
      break;
    default:
      std::cerr << "Specified matcher type is not supported...." << std::endl;
      std::exit(0);
    }

  return ptr;
}

}
}