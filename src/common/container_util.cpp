
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
      const common::vec1d<cv::KeyPoint>& key_point_list_train,
      const common::vec1d<cv::KeyPoint>& key_point_list_query,
      common::vec1d<T>& aligned_T_list_train,
      common::vec1d<T>& aligned_T_list_query,
      CopyFunctor copyFunc) {

  aligned_T_list_train.clear();
  aligned_T_list_train.reserve(matches.size());
  aligned_T_list_query.clear();
  aligned_T_list_query.reserve(matches.size());

  using constitr = common::vec1d<cv::DMatch>::const_iterator;
  for (constitr citr = matches.cbegin();
       citr != matches.cend();
       citr++) {
    assert(citr->trainIdx < key_point_list_train.size());
    aligned_T_list_train.push_back(copyFunc(key_point_list_train[citr->trainIdx]));
    assert(citr->queryIdx < key_point_list_query.size());
    aligned_T_list_query.push_back(copyFunc(key_point_list_query[citr->queryIdx]));
  }
}

template <typename A, typename B, typename CopyFunctor>
void convert_A_list_to_B_list(
      const common::vec1d<A>& A_list,
      common::vec1d<B>& B_list,
      CopyFunctor copyFunc) {

  B_list.clear();
  B_list.reserve(A_list.size());
  for (typename common::vec1d<A>::const_iterator citr = A_list.cbegin();
       citr != A_list.cend();
       citr++) {
    B_list.push_back(copyFunc(*citr));
  }

}

}

namespace simple_sfm {
namespace common {
namespace container_util {

void convert_key_point_list_to_point2f_list(
      const common::vec1d<cv::KeyPoint>& key_point_list,
      common::vec1d<cv::Point2f>& point2f_list) {

  auto copyFunc = [](const cv::KeyPoint& a) -> cv::Point2f { return a.pt; };
  convert_A_list_to_B_list<cv::KeyPoint, cv::Point2f>(
    key_point_list,
    point2f_list,
    copyFunc);
}

void convert_cloud_point_list_to_point3f_list(
      const common::vec1d<common::CloudPoint>& cloud_point_list,
      common::vec1d<cv::Point3f>& point3f_list) {

  auto copyFunc = [](const common::CloudPoint& p) -> cv::Point3f { return p.pt.coord; };
  convert_A_list_to_B_list<common::CloudPoint, cv::Point3f>(
    cloud_point_list,
    point3f_list,
    copyFunc);
}

void convert_cloud_point_list_to_point3d_list(
      const common::vec1d<common::CloudPoint>& cloud_point_list,
      common::vec1d<cv::Point3d>& point3d_list) {

  auto copyFunc = [](const common::CloudPoint& p) -> cv::Point3d { return p.pt.coord; };
  convert_A_list_to_B_list<common::CloudPoint, cv::Point3d>(
    cloud_point_list,
    point3d_list,
    copyFunc);
}

void convert_point3d_w_reperr_list_to_point3f_list(
      const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
      common::vec1d<cv::Point3f>& point3f_list) {

  auto copyFunc = [](const common::Point3dWithRepError& p) -> cv::Point3f { return p.coord; };
  convert_A_list_to_B_list<common::Point3dWithRepError, cv::Point3f>(
    point3d_w_reperr_list,
    point3f_list,
    copyFunc);
}

void convert_point3d_w_reperr_list_to_point3d_list(
      const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
      common::vec1d<cv::Point3d>& point3d_list) {

  auto copyFunc = [](const common::Point3dWithRepError& p) -> cv::Point3d { return p.coord; };
  convert_A_list_to_B_list<common::Point3dWithRepError, cv::Point3d>(
    point3d_w_reperr_list,
    point3d_list,
    copyFunc);
}

void convert_point3d_w_reperr_list_to_cloud_point_list(
      int image_num,
      const common::vec1d<common::Point3dWithRepError>& point3d_w_reperr_list,
      common::vec1d<common::CloudPoint>& cloud_point_list) {

  /*
  auto copyFunc = [=](const common::Point3dWithRepError& p) -> common::CloudPoint {
      common::CloudPoint cp(image_num);
      cp.pt = p;
      return cp; 
    };
  */

  struct CopyFunctor {

    CopyFunctor(int num) {
      image_num = num;
    }

    common::CloudPoint operator()(const common::Point3dWithRepError& p) {
      common::CloudPoint cp(image_num);
      cp.pt = p;
      return cp; 
    }
    int image_num;
  };

  CopyFunctor copyFunctor(image_num);

  convert_A_list_to_B_list<common::Point3dWithRepError, common::CloudPoint, CopyFunctor>(
    point3d_w_reperr_list,
    cloud_point_list,
    copyFunctor);
}

void create_key_point_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list_train,
      const common::vec1d<cv::KeyPoint>& key_point_list_query,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list_train,
      common::vec1d<cv::KeyPoint>& aligned_key_point_list_query){

  auto copyFunc = [](const cv::KeyPoint& a) -> cv::KeyPoint { return a;};
  convert_T_list_aligned_with_matches<cv::KeyPoint>(
    matches, key_point_list_train, key_point_list_query, 
    aligned_key_point_list_train, aligned_key_point_list_query,
    copyFunc);

}

void create_point2f_list_aligned_with_matches(
      const common::vec1d<cv::DMatch>& matches,
      const common::vec1d<cv::KeyPoint>& key_point_list_train,
      const common::vec1d<cv::KeyPoint>& key_point_list_query,
      common::vec1d<cv::Point2f>& aligned_point2f_list_train,
      common::vec1d<cv::Point2f>& aligned_point2f_list_query) {

  auto copyFunctor = [](const cv::KeyPoint& a) -> cv::Point2f {return a.pt;};
  convert_T_list_aligned_with_matches<cv::Point2f>(
    matches, key_point_list_train, key_point_list_query, 
    aligned_point2f_list_train, aligned_point2f_list_query,
    copyFunctor);

}

}
}
}