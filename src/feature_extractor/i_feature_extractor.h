

#ifndef I_FEATURE_EXTRACTOR_H
#define I_FEATURE_EXTRACTOR_H

// OpenCV
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace feature_extractor {

class IFeatureExtractor {
public:

  template <typename Value>
  using vec1d = simple_sfm::common::vec1d<Value>;

  template <typename Value>
  using vec2d = simple_sfm::common::vec2d<Value>;

  IFeatureExtractor()
  {}

  virtual ~IFeatureExtractor()
  {}

  virtual void detectAndCompute(
          const cv::Mat& image,
          vec1d<cv::KeyPoint>& key_points,
          cv::Mat& descriptors
          ) = 0;

  virtual void detectAndCompute(
          const vec1d<cv::Mat>& images,
          vec2d<cv::KeyPoint>& key_points,
          vec1d<cv::Mat>& descriptors
          ) = 0;
};

}
}

#endif
