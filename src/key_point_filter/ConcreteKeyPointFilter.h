
#ifndef CONCRETE_KEY_POINT_FILTER_H
#define CONCRETE_KEY_POINT_FILTER_H

// System
#include <assert.h>

// OpenCV
#include <opencv2/core.hpp>

// STL
#include <memory>

// Original
#include <key_point_filter/IKeyPointFilter.h>
#include <common/container.h>

using namespace std;

namespace simple_sfm {
namespace key_point_filter {

inline void align_point_wrt_match(
                  const common::vec1d<cv::KeyPoint>& key_point1,
                  const common::vec1d<cv::KeyPoint>& key_point2,
                  const common::vec1d<cv::DMatch>& match,
                  common::vec1d<cv::KeyPoint>& aligned_key_point1,
                  common::vec1d<cv::KeyPoint>& aligned_key_point2
                  ) {
  for (int i = 0; i < match.size(); i++) {
    assert(match[i].queryIdx < key_point1.size());
    aligned_key_point1.push_back(key_point1[match[i].queryIdx]);
    assert(match[i].trainIdx < key_point2.size());
    aligned_key_point2.push_back(key_point2[match[i].trainIdx]);
  }
}

inline void convert_keypoint_2_point2f(
                  const common::vec1d<cv::KeyPoint>& key_point,
                  common::vec1d<cv::Point2f>& point2f) {
  point2f.resize(key_point.size());
  for (int i = 0; i < key_point.size(); i++) {
    point2f[i] = key_point[i].pt;
  }
}

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

struct FMatKeyPointFilterInternalStorage;
class FMatKeyPointFilter : public IKeyPointFilter {
using Storage = FMatKeyPointFilterInternalStorage;
public:

  FMatKeyPointFilter();

  virtual ~FMatKeyPointFilter();

  virtual void run(const vec2d<cv::KeyPoint>& key_point_list,
                   const common::match_matrix& original_matrix,
                   common::match_matrix& new_matrix);

  virtual void filterKeyPoint(
                   const vec1d<cv::KeyPoint>& key_point1,
                   const vec1d<cv::KeyPoint>& key_point2,
                   const vec1d<cv::DMatch>& original_match,
                   vec1d<cv::DMatch>& new_match,
                   cv::Mat& calc_result);

private:
  std::unique_ptr<FMatKeyPointFilterInternalStorage> m_intl;
};

struct HomographyKeyPointFilterInternalStorage;
class HomographyKeyPointFilter : public IKeyPointFilter {
using Storage = HomographyKeyPointFilterInternalStorage;
public:
  HomographyKeyPointFilter();

  virtual ~HomographyKeyPointFilter();  

  virtual void run(const vec2d<cv::KeyPoint>& key_point_list,
                   const common::match_matrix& original_matrix,
                   common::match_matrix& new_matrix);

  virtual void filterKeyPoint(
                   const vec1d<cv::KeyPoint>& key_point1,
                   const vec1d<cv::KeyPoint>& key_point2,
                   const vec1d<cv::DMatch>& original_match,
                   vec1d<cv::DMatch>& new_match,
                   cv::Mat& calc_result);

private:
  std::unique_ptr<HomographyKeyPointFilterInternalStorage> m_intl;
};

}
}

#endif // CONCRETE_KEY_POINT_FILTER_H
