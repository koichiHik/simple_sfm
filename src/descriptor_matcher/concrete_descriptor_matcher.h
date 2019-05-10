
#ifndef CONCRETE_DESCRIPTOR_MATCHER_H
#define CONCRETE_DESCRIPTOR_MATCHER_H

// STL
#include <memory>

// OpenCV
#include <opencv2/gpu/gpu.hpp>

// Original
#include <descriptor_matcher/i_descriptor_matcher.h>

namespace simple_sfm {
namespace descriptor_matcher {

inline std::vector<cv::DMatch> 
flipMatches(const common::vec1d<cv::DMatch>& matches) {
  std::vector<cv::DMatch> flip;
  for (common::vec1d<cv::DMatch>::const_iterator citr = matches.cbegin();
       citr != matches.cend();
       citr++) {
    flip.push_back(*citr);
    std::swap(flip.back().queryIdx, flip.back().trainIdx);
  }
  return flip;
}

struct GPUBruteForceMatcherWithRatioCheckInternalStorage;
class GPUBruteForceMatcherWithRatioCheck : public IDescriptorMatcher {

public:

  GPUBruteForceMatcherWithRatioCheck();

  virtual ~GPUBruteForceMatcherWithRatioCheck();

  virtual void createMatchingMatrix(
                  const vec1d<cv::Mat>& descriptor_list,
                  common::match_matrix& match_matrix
                  );

private:
  virtual void match(
                  const cv::Mat& query_descriptor,
                  const cv::Mat& train_descriptor,
                  vec2d<cv::DMatch>& matches
                  );

private:
  std::unique_ptr<GPUBruteForceMatcherWithRatioCheckInternalStorage> m_intl;
};

struct GPUBruteForceMatcherInternalStorage;
class GPUBruteForceMatcher : public IDescriptorMatcher {

public:

  GPUBruteForceMatcher();

  virtual ~GPUBruteForceMatcher();

  virtual void createMatchingMatrix(
                  const vec1d<cv::Mat>& descriptor_list,
                  common::match_matrix& match_matrix
                  );

private:
  virtual void match(
                  const cv::Mat& query_descriptor,
                  const cv::Mat& train_descriptor,
                  vec2d<cv::DMatch>& matches
                  );

private:
  std::unique_ptr<GPUBruteForceMatcherInternalStorage> m_intl;
};


}
}

#endif