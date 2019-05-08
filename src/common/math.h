
#ifndef MATH_H
#define MATH_H

// System
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

namespace simple_sfm {
namespace common {
namespace math {

static const double VALID_ZERO_THRESH = 1e-06;

inline bool 
check_R_validity(cv::Matx33d& R) {
  double val = std::abs(cv::determinant(R) - 1.0);
  #if 0
  std::cout << "Calculated Value : " << val << std::endl;
  #endif
  return val < VALID_ZERO_THRESH;
}

inline bool 
check_R_validity(cv::Matx33f& R) {
  double val = std::abs(cv::determinant(R) - 1.0);
  #if 0
  std::cout << "Calculated Value : " << val << std::endl;
  #endif
  return val < VALID_ZERO_THRESH;
}

inline bool
check_E_validity(cv::Matx33d& E) {
  double val = std::abs(cv::determinant(E));
  #if 0
  std::cout << "Calculated Value : " << val << std::endl;
  #endif
  return val < VALID_ZERO_THRESH;
}

inline bool
check_E_validity(cv::Matx33f& E) {
  double val = std::abs(cv::determinant(E));
  #if 0
  std::cout << "Calculated Value : " << val << std::endl;
  #endif
  return val < VALID_ZERO_THRESH;
}

}
}
}



#endif // MATH_H