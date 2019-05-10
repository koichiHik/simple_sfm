
// System
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <key_point_filter/i_key_point_filter_factory.h>
#include <key_point_filter/concrete_key_point_filter.h>

namespace simple_sfm {
namespace key_point_filter {

cv::Ptr<IKeyPointFilter>
IKeyPointFilterFactory::
  createKeyPointFilter(FilterType type) {

    cv::Ptr<IKeyPointFilter> ptr;
    switch(type) {
    case FilterType::F_MAT:
      ptr = new FMatKeyPointFilter();
      break;
    case FilterType::HOMOGRAPHY:
      ptr = new HomographyKeyPointFilter();
      break;
    default:
      std::cerr << "Specified filter type is not supported...." << std::endl;
      std::exit(0);
    }

  return ptr;
}




}
}