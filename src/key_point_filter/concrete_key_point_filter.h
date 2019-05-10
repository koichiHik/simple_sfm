
#ifndef CONCRETE_KEY_POINT_FILTER_H
#define CONCRETE_KEY_POINT_FILTER_H

// System
#include <assert.h>

// OpenCV
#include <opencv2/core.hpp>

// STL
#include <memory>

// Original
#include <key_point_filter/i_key_point_filter.h>
#include <common/container.h>

using namespace std;

namespace simple_sfm {
namespace key_point_filter {

inline std::vector<cv::DMatch> 
flipMatches(const common::vec1d<cv::DMatch>& matches) {
  std::vector<cv::DMatch> flip;
  for (common::vec1d<cv::DMatch>::const_iterator citr = matches.cbegin();
       citr != matches.cend();
       citr++) {
    flip.push_back(*citr);
    std::swap(flip.back().trainIdx, flip.back().queryIdx);
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
