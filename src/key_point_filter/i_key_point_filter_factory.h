
#ifndef I_KEY_POINT_FILTER_FACTORY_H
#define I_KEY_POINT_FILTER_FACTORY_H

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <key_point_filter/i_key_point_filter.h>

namespace simple_sfm {
namespace key_point_filter {

enum FilterType : int {
  F_MAT,
  HOMOGRAPHY,
};

class IKeyPointFilterFactory {
public:
  static cv::Ptr<IKeyPointFilter>
      createKeyPointFilter(FilterType type);
};

}  
}
#endif // I_KEY_POINT_FILTER_FACTORY_H
