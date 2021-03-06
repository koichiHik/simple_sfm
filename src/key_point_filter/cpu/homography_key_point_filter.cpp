

// OpenCV
#include <opencv2/calib3d.hpp>

// Original
#include <common/container.h>
#include <common/container_util.h>
#include <key_point_filter/concrete_key_point_filter.h>

namespace simple_sfm {
namespace key_point_filter {

struct HomographyKeyPointFilterInternalStorage {
static const float REPROJ_ERR_SCALING;
};

const float HomographyKeyPointFilterInternalStorage
  ::REPROJ_ERR_SCALING = 0.004;

HomographyKeyPointFilter::HomographyKeyPointFilter() :
  m_intl(new HomographyKeyPointFilterInternalStorage())
{}

HomographyKeyPointFilter::~HomographyKeyPointFilter()
{}

void HomographyKeyPointFilter::run(
                   const vec2d<cv::KeyPoint>& key_point_list,
                   const common::match_matrix& original_matrix,
                   common::match_matrix& new_matrix) {

  std::cout << "HomographyKeyPointFilter" << std::endl;

  new_matrix.clear();
  for (size_t query_idx = 0; query_idx < key_point_list.size() - 1; query_idx++) {
    for (size_t train_idx = query_idx + 1; train_idx < key_point_list.size(); train_idx++) {

      std::pair<size_t, size_t> key = std::make_pair(train_idx, query_idx);
      const common::vec1d<cv::DMatch> original_match = common::getMapValue(original_matrix, key);
      common::vec1d<cv::DMatch> new_match;
      cv::Mat tmp;
      filterKeyPoint(
          key_point_list[train_idx],
          key_point_list[query_idx],
          original_match,
          new_match,
          tmp);

      new_matrix[key] = new_match;
      vec1d<cv::DMatch> flipped_match = flipMatches(new_match);
      new_matrix[std::make_pair(query_idx, train_idx)] = flipped_match;

      std::cout << "HomographyKeyPointFilter : Match (" << query_idx << ", " << train_idx << ")"
      << " shrinked " << new_match.size() << " / " << original_match.size() << std::endl;
    }
  }
}

void HomographyKeyPointFilter::filterKeyPoint(
                   const vec1d<cv::KeyPoint>& key_point_train,
                   const vec1d<cv::KeyPoint>& key_point_query,
                   const vec1d<cv::DMatch>& original_match,
                   vec1d<cv::DMatch>& new_match,
                   cv::Mat& calc_result) {

  vec1d<cv::KeyPoint> aligned_key_point_train, aligned_key_point_query;

  // Align keypoint wrt match.
  common::container_util::create_key_point_list_aligned_with_matches(
    original_match, key_point_train, key_point_query,
    aligned_key_point_train, aligned_key_point_query);

  vec1d<unsigned char> status(aligned_key_point_train.size());
  
  // Calculate Homography.
  {
    vec1d<cv::Point2f> aligned_point2f_train, aligned_point2f_query;
    common::container_util::convert_key_point_list_to_point2f_list(
      aligned_key_point_train, aligned_point2f_train);
    common::container_util::convert_key_point_list_to_point2f_list(
      aligned_key_point_query, aligned_point2f_query);
      
    double minVal, maxVal;
    cv::minMaxIdx(aligned_point2f_train, &minVal, &maxVal);
    calc_result = cv::findHomography(
            aligned_point2f_train,
            aligned_point2f_query,
            status,
            cv::RANSAC,
            Storage::REPROJ_ERR_SCALING * maxVal);
  }

  // Create new match based on "status"
  new_match.clear();
  {
    for (size_t i = 0; i < status.size(); i++) {
      // If status = 0, ignore it.
      if (status[i]) {
        new_match.push_back(original_match[i]);
      }
    }
  }
}

}
}