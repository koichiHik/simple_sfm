
#ifndef CONTAINER_H
#define CONTAINER_H

// System
#include <iostream>

// OpenCV
#include <opencv2/features2d.hpp> 

// STL
#include <map>
#include <vector>

namespace simple_sfm {
namespace common {

template <typename Value>
using vec1d = std::vector<Value>;

template <typename Value>
using vec2d = std::vector<vec1d<Value> >;

template <typename Value>
using vec3d = std::vector<vec2d<Value> >;

template <typename Value>
using matching_map = std::map<std::pair<size_t, size_t>, Value>;

using match_matrix = matching_map<vec1d<cv::DMatch> >;

template <typename Key, typename Value>
const Value& getMapValue(const std::map<Key, Value>& map, Key key) {
  typename std::map<Key, Value>::const_iterator itr = map.find(key);
  if (itr == map.cend()) {
    std::cerr << "Failed to extract map value." << std::endl;
    std::exit(0);
  }
  return itr->second;
}

struct CamIntrinsics {
  cv::Matx33d K;
  cv::Matx33d Kinv;
  cv::Mat_<double> distortion_coeff;
};

struct Point3dWithRepError {

  Point3dWithRepError() :
    valid(false), 
    reprojection_err(0.0),
    coord(0.0, 0.0, 0.0)
  {}

  bool valid;
  double reprojection_err;
  cv::Point3d coord;
};

struct CloudPoint {

  CloudPoint(int img_num) :
    pt(), idx_in_img(img_num, -1)
  {}

  Point3dWithRepError pt;
  std::vector<int> idx_in_img;
};

}
}

#endif //CONTAINER_H
