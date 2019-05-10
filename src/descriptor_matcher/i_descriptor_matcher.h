
#ifndef I_FEATURE_MATCHER_CALCULATOR_H
#define I_FEATURE_MATCHER_CALCULATOR_H

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace descriptor_matcher {

class IDescriptorMatcher {
public:

  template <typename Value>
  using vec1d = simple_sfm::common::vec1d<Value>;

  template <typename Value>
  using vec2d = simple_sfm::common::vec2d<Value>;

  template <typename Value>
  using vec3d = simple_sfm::common::vec3d<Value>;

  IDescriptorMatcher()
  {}

  virtual ~IDescriptorMatcher()
  {}

  virtual void createMatchingMatrix(
                  const vec1d<cv::Mat>& descriptor_list,
                  common::match_matrix& match_matrix
                  ) = 0;

private:

  virtual void match(
                  const cv::Mat& query_descriptor,
                  const cv::Mat& train_descriptor,
                  vec2d<cv::DMatch>& matches
                  ) = 0;
};

}
}

#endif
