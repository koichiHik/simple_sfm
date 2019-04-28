
#ifndef GPU_SUFT_FEATURE_CALCULATOR_H
#define GPU_SUFT_FEATURE_CALCULATOR_H

// OpenCV
#include <opencv2/core.hpp>

// STL
#include <memory>

// Original
#include <feature/IFeatureCalculator.h>
#include <common/container.h>

namespace simple_sfm {
namespace feature {

struct GPUSurfFeatureCalculatorInternalStorage;
class GPUSurfFeatureCalculator : public IFeatureCalculator {

public:

  GPUSurfFeatureCalculator();

  virtual ~GPUSurfFeatureCalculator();

  virtual void detectAndComputeForSingleImg(
          const cv::Mat& image,
          vec1d<cv::KeyPoint>& key_points,
          cv::Mat& descriptors
  );

  virtual void detectAndComputeForMultipleImgs(
          const vec1d<cv::Mat>& images,
          vec2d<cv::KeyPoint>& key_points,
          vec1d<cv::Mat>& descriptors
  );

private:
  std::unique_ptr<GPUSurfFeatureCalculatorInternalStorage> m_intl;
};

}
}

#endif