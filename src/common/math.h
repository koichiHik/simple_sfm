
#ifndef MATH_H
#define MATH_H

// OpenCV
#include <opencv2/core.hpp>

namespace simple_sfm {
namespace common {
namespace math {

static const double VALID_ZERO_THRESH = 1e-07;

inline bool 
check_R_validity(cv::Matx33d& R) {
  return std::abs(cv::determinant(R) - 1.0) < VALID_ZERO_THRESH;
}

inline bool
check_E_validity(cv::Matx33d& E) {
  return std::abs(cv::determinant(E)) < VALID_ZERO_THRESH;
}

}
}
}



#endif // MATH_H