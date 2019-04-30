
#ifndef CONTAINER_UTIL_H
#define CONTAINER_UTIL_H

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Original
#include <common/container.h>

namespace simple_sfm {
namespace common {
namespace container_util {

void convert_key_point_list_to_point2f_list(
      const common::vec1d<cv::KeyPoint>& key_point_list,
      common::vec1d<cv::Point2f>& point2f_list);

void create_key_point_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list1,
      const common::vec1d<cv::KeyPoint>& key_point_list2,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list1,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list2);

void create_point2f_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list1,
      const common::vec1d<cv::KeyPoint>& key_point_list2,
      common::vec1d<cv::Point2f>& aligned_point2f_list1,
      common::vec1d<cv::Point2f>& aligned_point2f_list2);

}
}
}

#endif // CONTAINER_UTIL_H
