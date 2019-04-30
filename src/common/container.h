
#ifndef CONTAINER_H
#define CONTAINER_H

// System
#include <iostream>

// OpenCV
#include <opencv2/features2d.hpp> 

// STL
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

}
}

#endif //CONTAINER_H
