

#ifndef I_KEY_POINT_FILTER_H
#define I_KEY_POINT_FILTER_H

// OpenCV
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace key_point_filter {

class IKeyPointFilter {
public:

  template <typename Value>
  using vec1d = simple_sfm::common::vec1d<Value>;

  template <typename Value>
  using vec2d = simple_sfm::common::vec2d<Value>;

  IKeyPointFilter()
  {}

  virtual ~IKeyPointFilter()
  {}
  
  virtual void run(const vec2d<cv::KeyPoint>& key_point_list,
                   const common::match_matrix& original_matrix,
                   common::match_matrix& naw_matrix) = 0;

  virtual void filterKeyPoint(
                   const vec1d<cv::KeyPoint>& key_point1,
                   const vec1d<cv::KeyPoint>& key_point2,
                   const vec1d<cv::DMatch>& original_match,
                   vec1d<cv::DMatch>& new_match,
                   cv::Mat& calc_result) = 0;
};

}
}

#endif // I_KEY_POINT_FILTER_H
