
// STL
#include <algorithm>

// Original
#include <common/container.h>
#include <common/container_util.h>

namespace {

using namespace simple_sfm;

template <typename T, typename CopyFunctor>
void convert_T_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list1,
      const common::vec1d<cv::KeyPoint>& key_point_list2,
      common::vec1d<T>& aligned_T_list1,
      common::vec1d<T>& aligned_T_list2,
      CopyFunctor copyFunc) {

  aligned_T_list1.clear();
  aligned_T_list1.reserve(matches.size());
  aligned_T_list2.clear();
  aligned_T_list2.reserve(matches.size());

  using constitr = common::vec1d<cv::DMatch>::const_iterator;
  for (constitr citr = matches.cbegin();
       citr != matches.cend();
       citr++) {
    assert(citr->queryIdx < key_point_list1.size());
    aligned_T_list1.push_back(copyFunc(key_point_list1[citr->queryIdx]));
    assert(citr->trainIdx < key_point_list2.size());
    aligned_T_list2.push_back(copyFunc(key_point_list2[citr->trainIdx]));
  }
}

}

namespace simple_sfm {
namespace common {
namespace container_util {

void convert_key_point_list_to_point2f_list(
      const common::vec1d<cv::KeyPoint>& key_point_list,
      common::vec1d<cv::Point2f>& point2f_list) {

  point2f_list.resize(key_point_list.size());
  std::transform(
    key_point_list.begin(), 
    key_point_list.end(),
    point2f_list.begin(),
    [](const cv::KeyPoint& a) -> cv::Point2f { return a.pt; });

}

void create_key_point_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list1,
      const common::vec1d<cv::KeyPoint>& key_point_list2,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list1,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list2){

  auto copyFunctor = [](const cv::KeyPoint& a) -> cv::KeyPoint { return a;};
  convert_T_list_aligned_with_matches<cv::KeyPoint>(
    matches, key_point_list1, key_point_list2, 
    aligned_key_point_list1, aligned_key_point_list1,
    copyFunctor);

}

void create_point2f_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list1,
      const common::vec1d<cv::KeyPoint>& key_point_list2,
      common::vec1d<cv::Point2f>& aligned_point2f_list1,
      common::vec1d<cv::Point2f>& aligned_point2f_list2) {

  auto copyFunctor = [](const cv::KeyPoint& a) -> cv::Point2f {return a.pt;};
  convert_T_list_aligned_with_matches<cv::Point2f>(
    matches, key_point_list1, key_point_list2, 
    aligned_point2f_list1, aligned_point2f_list2,
    copyFunctor);

}

}
}
}