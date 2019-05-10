
#ifndef CONCRETE_FEATURE_EXTRACTOR_H
#define CONCRETE_FEATURE_EXTRACTOR_H

// OpenCV
#include <opencv2/core.hpp>

// STL
#include <memory>

// Original
#include <feature_extractor/i_feature_extractor.h>
#include <common/container.h>

namespace simple_sfm {
namespace feature_extractor {

struct GPUSurfFeatureExtractorInternalStorage;
class GPUSurfFeatureExtractor : public IFeatureExtractor {

public:

  GPUSurfFeatureExtractor();

  virtual ~GPUSurfFeatureExtractor();

  virtual void detectAndCompute(
          const cv::Mat& image,
          vec1d<cv::KeyPoint>& key_points,
          cv::Mat& descriptors
  );

  virtual void detectAndCompute(
          const vec1d<cv::Mat>& images,
          vec2d<cv::KeyPoint>& key_points,
          vec1d<cv::Mat>& descriptors
  );

private:
  std::unique_ptr<GPUSurfFeatureExtractorInternalStorage> m_intl;
};

}
}

#endif