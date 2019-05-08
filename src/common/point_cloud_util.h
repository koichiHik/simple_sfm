
#ifndef POINT_CLOUD_UTIL_H
#define POINT_CLOUD_UTIL_H

// STL
#include <set>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <common/point_cloud_util.h>
#include <common/container.h>

namespace simple_sfm {
namespace common {

int find_best_matching_img_with_current_cloud(
  const std::set<size_t>& processed_view,
  const std::set<size_t>& adopted_view,  
  const common::match_matrix& matches,
  const common::vec2d<cv::Point2f>& img_pnts,
  const common::vec1d<CloudPoint>& cloud,
  common::vec1d<cv::Point2f>& corresp_2d_pnts,
  common::vec1d<cv::Point3f>& corresp_3d_pnts  
);

void extract_corresp_2d3dpnts_between_img_pnts_and_point_cloud(
  size_t query_img_idx,
  const std::set<size_t>& processed_view,
  const common::match_matrix& matches,
  const common::vec2d<cv::Point2f>& img_pnts,
  const common::vec1d<CloudPoint>& cloud,
  common::vec1d<cv::Point2f>& corresp_2d_pnts,
  common::vec1d<cv::Point3f>& corresp_3d_pnts);

}
}

#endif // POINT_CLOUD_UTIL_H
