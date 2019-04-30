
// System
#include <iostream>

// OpenCV
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

// Original
#include <common/container.h>
#include <key_point_filter/ConcreteKeyPointFilter.h>

namespace simple_sfm {
namespace key_point_filter {

struct FMatKeyPointFilterInternalStorage {
static const float REPROJ_ERR_SCALING;
static const float RANSAC_CONFIDENCE;
};

const float FMatKeyPointFilterInternalStorage
  ::REPROJ_ERR_SCALING = 0.00006;
const float FMatKeyPointFilterInternalStorage
  ::RANSAC_CONFIDENCE = 0.99;

FMatKeyPointFilter::FMatKeyPointFilter() :
  m_intl(new FMatKeyPointFilterInternalStorage())  
{}

FMatKeyPointFilter::~FMatKeyPointFilter() 
{}

void FMatKeyPointFilter::run(
                   const vec2d<cv::KeyPoint>& key_point_list,
                   const common::match_matrix& original_matrix,
                   common::match_matrix& new_matrix) {

  std::cout << std::endl << "FMatKeyPointFilter" << std::endl;

  new_matrix.clear();
  for (size_t query_idx = 0; query_idx < key_point_list.size() - 1; query_idx++) {
    for (size_t train_idx = query_idx + 1; train_idx < key_point_list.size(); train_idx++) {

      std::pair<size_t, size_t> key = std::make_pair(query_idx, train_idx);
      const common::vec1d<cv::DMatch> original_match = common::getMapValue(original_matrix, key);
      common::vec1d<cv::DMatch> new_match;
      cv::Mat tmp;
      filterKeyPoint(
          key_point_list[query_idx],
          key_point_list[train_idx],
          original_match,
          new_match,
          tmp);

      new_matrix[key] = new_match;
      vec1d<cv::DMatch> flipped_match = flipMatches(new_match);
      new_matrix[std::make_pair(train_idx, query_idx)] = flipped_match;

      std::cout << "FMatKeyFilter : Match (" << query_idx << ", " << train_idx << ")"
      << " shrinked " << new_match.size() << " / " << original_match.size() << std::endl;
    }
  }
}

void FMatKeyPointFilter::filterKeyPoint(
                   const vec1d<cv::KeyPoint>& key_point1,
                   const vec1d<cv::KeyPoint>& key_point2,
                   const vec1d<cv::DMatch>& original_match,
                   vec1d<cv::DMatch>& new_match,
                   cv::Mat& calc_result) {

  vec1d<cv::KeyPoint> aligned_key_point1, aligned_key_point2;

  // Align keypoint wrt match.
  align_point_wrt_match(key_point1, key_point2, 
    original_match, aligned_key_point1, aligned_key_point2);

  vec1d<unsigned char> status(aligned_key_point1.size());
  
  // Calculate Fundamental Matrix.
  {
    vec1d<cv::Point2f> aligned_point2f_1, aligned_point2f_2;
    convert_keypoint_2_point2f(aligned_key_point1, aligned_point2f_1);
    convert_keypoint_2_point2f(aligned_key_point2, aligned_point2f_2);
    
    double minVal, maxVal;
    cv::minMaxIdx(aligned_point2f_1, &minVal, &maxVal);
    calc_result = cv::findFundamentalMat(
            aligned_point2f_1, 
            aligned_point2f_2, 
            cv::FM_RANSAC, 
            Storage::REPROJ_ERR_SCALING * maxVal,
            Storage::RANSAC_CONFIDENCE, 
            status);
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