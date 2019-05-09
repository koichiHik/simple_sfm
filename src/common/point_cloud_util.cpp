

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
) {
  int best_img_idx = -1;
  common::vec1d<cv::Point2f> best_matching_point2d;
  common::vec1d<cv::Point3f> best_matching_point3d;

  for (size_t img_idx = 0; img_idx < img_pnts.size(); img_idx++) {

    // If already processed, skip.
    if (processed_view.find(img_idx) != processed_view.cend()) {
      continue;
    }

    common::vec1d<cv::Point3f> matching_point3d;
    common::vec1d<cv::Point2f> matching_point2d;

    extract_corresp_2d3dpnts_between_img_pnts_and_point_cloud(
      img_idx, processed_view, matches,
      img_pnts, cloud, matching_point2d, matching_point3d);

    if (best_matching_point3d.size() < matching_point3d.size()) {
      best_img_idx = img_idx;
      best_matching_point2d = matching_point2d;
      best_matching_point3d = matching_point3d;
    }

  }

  corresp_2d_pnts = best_matching_point2d;
  corresp_3d_pnts = best_matching_point3d;

  return best_img_idx;
}

void extract_corresp_2d3dpnts_between_img_pnts_and_point_cloud(
  size_t query_img_idx,
  const std::set<size_t>& processed_view,
  const common::match_matrix& matches,
  const common::vec2d<cv::Point2f>& img_pnts,
  const common::vec1d<CloudPoint>& cloud,
  common::vec1d<cv::Point2f>& corresp_2d_pnts,
  common::vec1d<cv::Point3f>& corresp_3d_pnts) {

  corresp_2d_pnts.clear();
  corresp_3d_pnts.clear();

  common::vec1d<int> pcloud_status(cloud.size(), 0);

  // Loop : Already processed images.
  for (std::set<size_t>::const_iterator citr = processed_view.cbegin();
       citr != processed_view.cend();
       citr++) {
    size_t train_img_idx = *citr;
    
    std::pair<size_t, size_t> key = std::make_pair(train_img_idx, query_img_idx);
    const common::vec1d<cv::DMatch>& match = common::getMapValue(matches, key);

    // Loop : Keypoint match vectors with the processed frame.
    for (common::vec1d<cv::DMatch>::const_iterator citr_match = match.cbegin();
         citr_match != match.cend();
         citr_match++) {
      int idx_in_train_img = citr_match->trainIdx;

      // Loop : Point in current point cloud..
      for (int cp_idx = 0; cp_idx < cloud.size(); cp_idx++) {
        if (cloud[cp_idx].idx_in_img[train_img_idx] == idx_in_train_img &&
            pcloud_status[cp_idx] == 0) {
          corresp_2d_pnts.push_back(img_pnts[query_img_idx][citr_match->queryIdx]);
          corresp_3d_pnts.push_back(cloud[cp_idx].pt.coord);

          // Avoid duplicated point insertion.
          pcloud_status[cp_idx] = 1;
          break;
        }
      }
    }
  }
}

}
}
